#!/usr/bin/env python3
# Fixed-bit ablation for OQG — testing side (mirrors test_mn.py).
# For each b in BIT_LIST, loads the index built by train_fixed_bit.py and
# sweeps efSearch; CSV schema = test_mn.py schema + BitPerDim (configured b)
# + actualBitPerDim (= 8 * num_subspaces / dataset original dim).
#
# Usage: python test_fixed_bit.py --ds gist
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

EDGE_NUM = 16   # filename tag; must match train_fixed_bit.py
repeat = 1

BIT_LIST = [0.5, 1, 2, 3]   # must match train_fixed_bit.py

topk = 100
CONFIG = {
    "m": 64,
    "efC": 600,

    "pq_dir":  "/home/cc/dataset/cb/fixed_bit",       # must match train_fixed_bit.py
    "index_dir": "/home/cc/dataset/index/GG/fixed_bit",

    "out_csv": os.path.join(SCRIPT_DIR, "test_fixed_bit.csv"),

    "max_workers": 1,
}

# same efSearch sweep as OQG's test_mn.py
efSList = [100, 110, 130, 135, 140, 145, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000]


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


def subspaces_for_bit(bit: float, dim: int) -> int:
    # 8 bits per PQ subspace; round half up (四舍五入), at least 1
    return max(1, int(bit * dim / 8.0 + 0.5))


def actual_bit_per_dim(num_subspaces: int, dim: int) -> float:
    return round(8.0 * num_subspaces / dim, 4)


def paths_for(cfg: dict, dataset: str, bit: float, num_subspaces: int, preprocess: str):
    pq_path = f"{cfg['pq_dir']}/{dataset}_b{bit}_8x{num_subspaces}_{preprocess}.npz"
    index_path = (f"{cfg['index_dir']}/{dataset}_b{bit}_{cfg['m']}_8x{num_subspaces}"
                  f"_{cfg['efC']}_{EDGE_NUM}.mnggindex")
    return pq_path, index_path


def test_one_dataset(dataset: str, cfg: dict, eList: list[int]) -> list[dict]:
    search_method = sms.get(dataset, 2)

    all_conf = get_datasets_config([dataset], None)[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"

    base0  = read_vecs(all_conf["base"]).astype(np.float32)
    query0 = read_vecs(all_conf["query"]).astype(np.float32)
    gt = read_vecs(all_conf["gt"])
    N, dim0 = base0.shape   # dataset original dim (used for actualBitPerDim)
    Q = query0.shape[0]

    results = []
    for bit in BIT_LIST:
        num_subspaces = subspaces_for_bit(bit, dim0)
        pq_path, index_path = paths_for(cfg, dataset, bit, num_subspaces, preprocess)

        if not os.path.exists(index_path) or not os.path.exists(pq_path):
            print(f"[SKIP] {dataset} b={bit}: missing {index_path} or {pq_path}")
            continue

        base = pad_to_mod(base0, num_subspaces)
        base = np.asarray(base, dtype=np.float32, order='C')
        query = pad_to_mod(query0, num_subspaces)
        query = np.ascontiguousarray(query, dtype=np.float32)
        dim = base.shape[1]

        npz = np.load(pq_path)
        if use_opq:
            opq_matrix = npz["opq_matrix"].T
            opq_matrix = np.ascontiguousarray(opq_matrix, dtype=np.float32)
        else:
            opq_matrix = None

        p = oqglib.MNGGIndex(index_path, base, num_subspaces, dim)
        del npz
        gc.collect()

        abpd = actual_bit_per_dim(num_subspaces, dim0)

        for ef_search in eList:
            assert ef_search >= topk

            num_refine = ef_search
            mem = get_max_resident_memory_gb()

            if search_method == 1:
                labels, latency = p.searchKNNPQ(opq_matrix, query, ef_search, topk, num_refine)
                for i in range(repeat - 1):
                    labels, cur_latency = p.searchKNNPQ(opq_matrix, query, ef_search, topk, num_refine)
                    latency = min(latency, cur_latency)
            elif search_method == 2:
                labels, latency = p.searchKNNPQ16(opq_matrix, query, ef_search, topk, num_refine)
                for i in range(repeat - 1):
                    labels, cur_latency = p.searchKNNPQ16(opq_matrix, query, ef_search, topk, num_refine)
                    latency = min(latency, cur_latency)
            else:
                raise ValueError(f"unknown search_method={search_method}")

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
                "BitPerDim": bit,
                "actualBitPerDim": abpd,
            }
            # flush immediately so a crash on a later bit does not lose rows
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


parser = argparse.ArgumentParser(description="OQG fixed-bit ablation testing")
parser.add_argument("--ds", type=str, required=True, help="ds")
args = parser.parse_args()

DATASETS = get_datasets_config([args.ds], mod=None).keys()


def main():
    cfg = deepcopy(CONFIG)
    out_csv = cfg["out_csv"]

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
            print(f"Saved {len(results)} rows to {out_csv}")
        else:
            print("No results.")


if __name__ == "__main__":
    main()
