#!/usr/bin/env python3
# CBCA d = D/4 sensitivity (testing side):
#   load the graph index built by train_cbca.py for a (D, d) configuration
#   and sweep efSearch, producing the same CSV schema as test_mn.py
#   (consumed by Evaluations/oqg_revision/CBCA_Sensitivity/analyze.py).
#
# Usage: python test_cbca.py --ds sift --batch 128 --edge 32
import os
import sys
import csv
import gc
import argparse
import numpy as np
import oqglib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OQG_EXAMPLE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, OQG_EXAMPLE_DIR)
from dataset import get_datasets_config, read_vecs
from dataset_config import suggested_subspaces, pqopq, sms

EF_CONSTRUCTION = 600
TOPK = 100
REPEAT = 5

EFS_LIST = [100, 110, 130, 135, 140, 145, 150, 170, 190, 200,
            250, 300, 400, 500, 600, 700, 800, 1000]


def get_index_class(batch: int, edge: int):
    legacy = {(64, 16): "MNGGIndex", (64, 64): "MNGGIndexE64"}
    name = legacy.get((batch, edge), f"MNGGIndexB{batch}E{edge}")
    cls = getattr(oqglib, name, None)
    if cls is None:
        raise RuntimeError(
            f"oqglib.{name} not found: recompile python_bindings with "
            f"(batchSize={batch}, numLevel0Edges={edge}) exposed")
    return cls


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


def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def test_one_dataset(dataset: str, batch: int, edge: int) -> list[dict]:
    m = batch
    efC = EF_CONSTRUCTION

    pq_dir = f"/home/cc/dataset/cb/batch{batch}edge{edge}"
    index_dir = f"/home/cc/dataset/index/GG/batch{batch}edge{edge}"

    # On very rare occasions, search_method=1 yields slightly better results.
    search_method = sms.get(dataset, 2)

    conf = get_datasets_config([dataset], None)[dataset]

    num_subspaces = suggested_subspaces[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"

    base = read_vecs(conf["base"]).astype(np.float32)
    base = pad_to_mod(base, num_subspaces)
    query = read_vecs(conf["query"]).astype(np.float32)
    query = pad_to_mod(query, num_subspaces)
    gt = read_vecs(conf["gt"])
    N, dim = base.shape
    Q = query.shape[0]

    pq_path = f"{pq_dir}/{dataset}_8x{num_subspaces}_{preprocess}.npz"
    index_path = f"{index_dir}/{dataset}_{m}_8x{num_subspaces}_{efC}_{edge}.mnggindex"

    if not os.path.exists(pq_path):
        raise FileNotFoundError(f"Not Found for PQ Files: {pq_path}")
    npz = np.load(pq_path)
    query = np.ascontiguousarray(query, dtype=np.float32)
    if use_opq:
        opq_matrix = npz["opq_matrix"].T
        opq_matrix = np.ascontiguousarray(opq_matrix, dtype=np.float32)
    else:
        opq_matrix = None

    base = np.asarray(base, dtype=np.float32, order='C')

    IndexCls = get_index_class(batch, edge)
    assert IndexCls.batchSize == batch and IndexCls.numLevel0Edges == edge
    p = IndexCls(index_path, base, num_subspaces, dim)

    del npz
    gc.collect()

    results = []
    for ef_search in EFS_LIST:
        assert ef_search >= TOPK

        num_refine = ef_search
        mem = get_max_resident_memory_gb()

        if search_method == 1:
            labels, latency = p.searchKNNPQ(opq_matrix, query, ef_search, TOPK, num_refine)
            for _ in range(REPEAT - 1):
                labels, cur_latency = p.searchKNNPQ(opq_matrix, query, ef_search, TOPK, num_refine)
                latency = min(latency, cur_latency)
        elif search_method == 2:
            labels, latency = p.searchKNNPQ16(opq_matrix, query, ef_search, TOPK, num_refine)
            for _ in range(REPEAT - 1):
                labels, cur_latency = p.searchKNNPQ16(opq_matrix, query, ef_search, TOPK, num_refine)
                latency = min(latency, cur_latency)
        else:
            raise ValueError(f"unknown search_method={search_method}")

        recall_k = compute_recall(labels.astype(np.int64), gt, TOPK)
        throughput = Q / latency
        print(f"[{dataset}] batch={batch} edge={edge} efS={ef_search} "
              f"recall={recall_k:.5f} qps={throughput:.1f}")
        results.append({
            "Dataset": dataset,
            "efSearch": ef_search,
            "Recall": recall_k,
            "QPS": throughput,
            "latency": latency,
            "MemGB": mem,
            "k": TOPK,
            "opq": use_opq,
            "NumSubspaces": num_subspaces,
            "NumBase": N,
            "NumQuery": Q
        })

    del p
    del base
    if use_opq:
        del opq_matrix
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="CBCA sensitivity testing")
    parser.add_argument("--ds", type=str, required=True, help="dataset name")
    parser.add_argument("--batch", type=int, required=True,
                        help="graph degree D (= batchSize)")
    parser.add_argument("--edge", type=int, required=True,
                        help="CBCA level-0 edge count d (= numLevel0Edges)")
    args = parser.parse_args()

    out_csv = os.path.join(SCRIPT_DIR, f"test_batch{args.batch}_edge{args.edge}.csv")

    rows = test_one_dataset(args.ds, args.batch, args.edge)
    for row in rows:
        append_row(out_csv, row)
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
