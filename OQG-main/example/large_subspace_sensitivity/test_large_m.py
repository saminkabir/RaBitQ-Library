#!/usr/bin/env python3
# Large-subspace sensitivity for OQG — testing side.
# For each M in M_LIST (128/256/512/1024 subspaces on GIST):
#   1. Saturation analysis: replicate the C++ 16-bit LUT quantization
#      (quantize16LUT -> _saturated -> _mean chain in oqg_multi.h, after the
#      subspace-major layout fix) in numpy, then measure the fraction of
#      (query, base point) pairs whose accumulated quantized distance exceeds
#      65535 — i.e. would wrap around in the uint16 accumulation kernel.
#   2. Recall/QPS sweep over efSearch via searchKNNPQ16 (16-bit accumulation).
# CSV schema = test_mn.py schema + NumSubspaces is the swept M
#            + SaturationRate / MeanQDistSum / P99QDistSum / MaxQDistSum.
#
# Usage: python test_large_m.py --ds gist
import os
import sys
import csv
import gc
import time
import argparse
import numpy as np
import oqglib
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OQG_EXAMPLE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, OQG_EXAMPLE_DIR)
from dataset import get_datasets_config, read_vecs
from dataset_config import pqopq, sms

EDGE_NUM = 16   # filename tag; must match train_large_m.py
repeat = 1

#M_LIST = [128, 256, 512, 1024]   # must match train_large_m.py

M_LIST = [768]

# saturation analysis sampling (population estimate over random pairs)
SAT_NUM_QUERIES = 100
SAT_NUM_POINTS  = 100_000
SAT_SEED        = 123
U16_MAX         = 65535

topk = 100
CONFIG = {
    "m": 64,
    "efC": 600,

    "pq_dir":  "/home/cc/dataset/cb/large_subspace",       # must match train_large_m.py
    "index_dir": "/home/cc/dataset/index/GG/large_subspace",

    "out_csv": os.path.join(SCRIPT_DIR, "test_large_m.csv"),

    "max_workers": 1,
}

# same efSearch sweep as OQG's test_mn.py
efSList = [100, 110, 130, 135, 140, 145, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 1000]


def get_max_resident_memory_gb() -> float:
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb / 1024 / 1024
    return 0.0


def pad_to_mod(arr: np.ndarray, mod: int, pad_value: float = 0.0) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("Only Support 2D array")
    N, D = arr.shape
    r = D % mod
    if r == 0:
        return arr
    pad_len = mod - r
    pad_block = np.full((N, pad_len), pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad_block], axis=1)


def compute_recall(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    rk = retrieved[:, :k]
    gk = ground_truth[:, :k]
    hits = (rk[:, :, None] == gk[:, None, :]).any(axis=2).sum(axis=1)
    return float(hits.sum() / (rk.shape[0] * k))


def paths_for(cfg: dict, dataset: str, num_subspaces: int, preprocess: str):
    pq_path = f"{cfg['pq_dir']}/{dataset}_sub{num_subspaces}_{preprocess}.npz"
    index_path = (f"{cfg['index_dir']}/{dataset}_sub{num_subspaces}_{cfg['m']}"
                  f"_{cfg['efC']}_{EDGE_NUM}.mnggindex")
    return pq_path, index_path


# ──────────────────────────────────────────────
# numpy replication of the 16-bit LUT quantization in oqg_multi.h
# (quantize16LUT -> quantize16LUT_saturated -> quantize16LUT_mean,
#  subspace-major layout, after the layout fix)
# ──────────────────────────────────────────────
def trim_trailing_zero_subspaces(lut: np.ndarray, eps: float = 1e-20) -> int:
    adj = lut.shape[0]
    while adj >= 1 and np.all(np.abs(lut[adj - 1]) <= eps):
        adj -= 1
    return adj


def _quantize16_mean(lut: np.ndarray, adj: int) -> np.ndarray:
    cols = lut[:adj]
    finite = np.isfinite(cols)
    colmin = np.where(finite, cols, np.inf).min(axis=1)
    colmean = np.where(finite, cols, 0.0).sum(axis=1) / lut.shape[1]
    thr = colmin * adj
    qmax = float(np.minimum(colmean, thr).sum())
    qmin = float(colmin.min())
    if not (qmax > qmin):
        return np.full(lut.shape, 128, dtype=np.uint16)
    a = max(abs(qmin), abs(qmax))
    if not (np.isfinite(a) and a > 0):
        a = 1.0
    x = np.clip(lut, -a, a)
    q = np.floor(x / a * 127.0 + 0.5) + 128
    q = np.clip(q, 0, 255)
    q[lut == 0] = 128
    q[~np.isfinite(lut)] = 128
    return q.astype(np.uint16)


def _quantize16_saturated(lut: np.ndarray, adj: int, kper: float = 0.9) -> np.ndarray:
    active = lut[:adj]
    vals = active[np.isfinite(active)]
    if vals.size == 0:
        return _quantize16_mean(lut, adj)
    qmin = float(vals.min())
    n = vals.size
    kth = min(n - 1, int(np.floor(kper * (n - 1))))
    qmax = float(np.partition(vals, kth)[kth])
    if (not (qmax > qmin)) or (qmax > qmin * 65525):
        return _quantize16_mean(lut, adj)
    scale = 255.0 / (qmax - qmin)
    bias = -qmin * scale
    x = np.clip(lut, qmin, qmax)
    q = np.floor(x * scale + bias + 0.5)
    q = np.clip(q, 0, 255)
    q[~np.isfinite(lut)] = 128
    return q.astype(np.uint16)


def quantize16_lut(lut: np.ndarray, adj: int) -> np.ndarray:
    vals = lut[np.isfinite(lut)]
    qmin, qmax = float(vals.min()), float(vals.max())
    if (not (qmax > qmin)) or (qmax > qmin * 65536):
        return _quantize16_saturated(lut, adj)
    a = max(abs(qmin), abs(qmax))
    if not (np.isfinite(a) and a > 0):
        a = 1.0
    x = np.clip(lut, -a, a)
    q = np.floor(x / a * 127.0 + 0.5) + 128
    q = np.clip(q, 0, 255)
    q[lut == 0] = 128
    return q.astype(np.uint16)


def saturation_analysis(queries_rot: np.ndarray, centroids: np.ndarray,
                        codes: np.ndarray) -> dict:
    """queries_rot: (Q, dim) already rotated/padded; centroids: (256, M, dsub);
    codes: (N, M) uint8. Returns saturation stats over sampled pairs."""
    rng = np.random.default_rng(SAT_SEED)
    Q, N = queries_rot.shape[0], codes.shape[0]
    M = centroids.shape[1]

    q_idx = rng.choice(Q, size=min(SAT_NUM_QUERIES, Q), replace=False)
    p_idx = rng.choice(N, size=min(SAT_NUM_POINTS, N), replace=False)
    codes_s = np.ascontiguousarray(codes[p_idx].T)          # (M, P)
    cent = np.ascontiguousarray(centroids.transpose(1, 0, 2))  # (M, 256, dsub)

    sat_pairs = 0
    total_pairs = 0
    sum_all = 0.0
    max_sum = 0
    p99s = []

    for qi in q_idx:
        qs = queries_rot[qi].reshape(M, -1)                 # (M, dsub)
        diff = cent - qs[:, None, :]
        lut = np.einsum('mcd,mcd->mc', diff, diff)          # (M, 256) float
        adj = trim_trailing_zero_subspaces(lut)
        q16 = quantize16_lut(lut, adj)                      # (M, 256) uint16

        dist = np.zeros(codes_s.shape[1], dtype=np.int64)
        for s in range(M):
            dist += q16[s][codes_s[s]]

        sat_pairs += int((dist > U16_MAX).sum())
        total_pairs += dist.size
        sum_all += float(dist.mean())
        max_sum = max(max_sum, int(dist.max()))
        p99s.append(float(np.percentile(dist, 99)))

    return {
        "SaturationRate": sat_pairs / total_pairs,
        "MeanQDistSum": sum_all / len(q_idx),
        "P99QDistSum": float(np.mean(p99s)),
        "MaxQDistSum": max_sum,
    }


# ──────────────────────────────────────────────
# testing
# ──────────────────────────────────────────────
def test_one_dataset(dataset: str, cfg: dict, eList: list[int]) -> list[dict]:
    search_method = sms.get(dataset, 2)
    assert search_method == 2, "this sensitivity targets the 16-bit accumulation path"

    all_conf = get_datasets_config([dataset], None)[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"

    base0  = read_vecs(all_conf["base"]).astype(np.float32)
    query0 = read_vecs(all_conf["query"]).astype(np.float32)
    gt = read_vecs(all_conf["gt"])
    N, dim0 = base0.shape
    Q = query0.shape[0]

    results = []
    for num_subspaces in M_LIST:
        pq_path, index_path = paths_for(cfg, dataset, num_subspaces, preprocess)

        if not os.path.exists(index_path) or not os.path.exists(pq_path):
            print(f"[SKIP] {dataset} M={num_subspaces}: missing {index_path} or {pq_path}")
            continue

        base = pad_to_mod(base0, num_subspaces)
        base = np.asarray(base, dtype=np.float32, order='C')
        query = pad_to_mod(query0, num_subspaces)
        query = np.ascontiguousarray(query, dtype=np.float32)
        dim = base.shape[1]

        npz = np.load(pq_path)
        centroids = npz["centroids"]        # (256, M, dsub)
        codes = npz["codes"]                # (N, M) uint8
        if use_opq:
            A = npz["opq_matrix"]           # (d_out, d_in)
            opq_matrix = np.ascontiguousarray(A.T, dtype=np.float32)
            queries_rot = np.ascontiguousarray(query @ A.T.astype(np.float32))
        else:
            opq_matrix = None
            queries_rot = query

        # 1. saturation analysis (query-side simulation of the u16 kernel)
        t0 = time.perf_counter()
        sat = saturation_analysis(queries_rot, centroids, codes)
        print(f"[SAT] {dataset} M={num_subspaces}: rate={sat['SaturationRate']:.6f} "
              f"mean={sat['MeanQDistSum']:.0f} p99={sat['P99QDistSum']:.0f} "
              f"max={sat['MaxQDistSum']} ({time.perf_counter()-t0:.1f}s)")

        del npz, centroids, codes, queries_rot
        gc.collect()

        # 2. recall / QPS sweep
        p = oqglib.MNGGIndex(index_path, base, num_subspaces, dim)
        gc.collect()

        for ef_search in eList:
            assert ef_search >= topk

            num_refine = ef_search
            mem = get_max_resident_memory_gb()

            labels, latency = p.searchKNNPQ16(opq_matrix, query, ef_search, topk, num_refine)
            for i in range(repeat - 1):
                labels, cur_latency = p.searchKNNPQ16(opq_matrix, query, ef_search, topk, num_refine)
                latency = min(latency, cur_latency)

            recall_k = compute_recall(labels.astype(np.int64), gt, topk)
            throughput = Q / latency
            row = {
                "Dataset": dataset,
                "efSearch": ef_search,
                "Recall": recall_k,
                "QPS": throughput,
                "latency": latency,
                "MemGB": mem,
                "k": topk,
                "opq": use_opq,
                "NumSubspaces": num_subspaces,
                "NumBase": int(N),
                "NumQuery": int(Q),
                "SaturationRate": sat["SaturationRate"],
                "MeanQDistSum": sat["MeanQDistSum"],
                "P99QDistSum": sat["P99QDistSum"],
                "MaxQDistSum": sat["MaxQDistSum"],
            }
            # flush immediately so a crash on a later M does not lose rows
            append_row(cfg["out_csv"], row)
            results.append(row)

        del p, base, query
        if use_opq:
            del opq_matrix
        gc.collect()

    del base0, query0
    gc.collect()

    return results


def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


parser = argparse.ArgumentParser(description="OQG large-subspace sensitivity testing")
parser.add_argument("--ds", type=str, default="gist", help="ds (default: gist)")
args = parser.parse_args()

DATASETS = get_datasets_config([args.ds], mod=None).keys()


def main():
    cfg = deepcopy(CONFIG)

    results = []
    with ProcessPoolExecutor(max_workers=cfg["max_workers"]) as ex:
        fut2ds = {
            ex.submit(test_one_dataset, ds, cfg, efSList): ds for ds in DATASETS
        }

        for fut in as_completed(fut2ds):
            ds = fut2ds[fut]
            try:
                # rows are already appended to the CSV inside test_one_dataset
                results.extend(fut.result())
            except Exception as e:
                print(f"[FAILED] {ds}: {e}")

        if results:
            print(f"Saved {len(results)} rows to {cfg['out_csv']}")
        else:
            print("No results.")


if __name__ == "__main__":
    main()
