#!/usr/bin/env python3
import os
import csv
import time
import numpy as np
import oqglib
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataset import get_datasets_config, read_vecs
from dataset_config import suggested_subspaces, pqopq, sms
import gc, time

EDGE_NUM = 16
repeat = 5
#assert repeat >= 5, "at least 5 to get stable performance"

def get_max_resident_memory_gb() -> float:
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb / 1024 / 1024
    return 0.0


def read_ivecs(fname: str) -> np.ndarray:
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def read_fvecs(fname: str) -> np.ndarray:
    return read_ivecs(fname).view("float32")

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



def test_one_dataset(dataset: str, cfg: dict, eList: list[float]) -> dict:

    m               = cfg["m"]
    pq_dir        = cfg["pq_dir"]
    index_dir       = cfg["index_dir"]
    efC             = cfg["efC"]
    search_method = sms.get(dataset, 2) # On very rare occasions, search_method=1 yields slightly better results.



    all_conf = get_datasets_config([dataset], None)[dataset]

    base_path  = all_conf["base"]
    query_path = all_conf["query"] 
    gt_path    = all_conf["gt"]  



    num_subspaces = suggested_subspaces[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"


    base  = read_vecs(base_path).astype(np.float32)
    base  = pad_to_mod(base, num_subspaces)
    query = read_vecs(query_path).astype(np.float32)
    query = pad_to_mod(query, num_subspaces)
    gt = read_vecs(gt_path)
    N, dim = base.shape
    Q = query.shape[0]


    pq_path        = f"{pq_dir}/{dataset}_8x{num_subspaces}_{preprocess}.npz"
    index_path     = f"{index_dir}/{dataset}_{m}_8x{num_subspaces}_{efC}_{EDGE_NUM}.mnggindex"


    if not os.path.exists(pq_path):
        raise FileNotFoundError(f"Not Found for PQ Files: {pq_path}")
    npz = np.load(pq_path)
    pq_codes      = npz["codes"]
    pq_centroids  = npz["centroids"]
    query = np.ascontiguousarray(query, dtype=np.float32)    
    if use_opq:
        opq_matrix = npz["opq_matrix"].T
        opq_matrix = np.ascontiguousarray(opq_matrix, dtype=np.float32)    
    else:
        opq_matrix = None

    id_mapping = np.arange(N, dtype=np.int32)
    base = np.asarray(base, dtype=np.float32, order='C')
    p = oqglib.MNGGIndex(index_path, base, num_subspaces, dim)

    # for i in range(10):
    #     p.printNodeNeighborsPQCodes(i)
    # exit()

    del npz
    del pq_codes
    del pq_centroids
    gc.collect()

    results = []
    for ef_search in eList:
        assert(ef_search >= topk)

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
        elif search_method == 3:
            labels, latency = p.searchKNNPQ16Batch(opq_matrix, query, ef_search, topk, num_refine)
            for i in range(repeat - 1):
                labels, cur_latency = p.searchKNNPQ16Batch(opq_matrix, query, ef_search, topk, num_refine)
                latency = min(latency, cur_latency)
        else:
            raise ValueError(f"unknown search_method={search_method}")


        recall_k  = compute_recall(labels.astype(np.int64), gt, topk)
        throughput = Q / latency
        results.append({
            "Dataset": dataset,
            "efSearch": ef_search,
            "Recall": recall_k,
            "QPS": throughput,
            "latency": latency,
            "MemGB": mem,
            "k": topk,
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



def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


topk = 100
CONFIG = {
    "m": 64,
    "efC": 600,

    "pq_dir":  "/home/cc/dataset/cb/batch64edge16", # dir path for PQ codebook
    "index_dir": "/home/cc/dataset/index/GG/batch64edge16", # dir path for graph index

    #"out_csv": f"CBCA_sensitivity/test_batch64_edge{EDGE_NUM}.csv",
    "out_csv": f"test_k{topk}_edge{EDGE_NUM}.csv",

    "max_workers": 1,
}




import argparse

parser = argparse.ArgumentParser(description="Demo for reading a parameter")
parser.add_argument("--ds", type=str, required=True, help="ds")
args = parser.parse_args()

DATASETS = get_datasets_config([args.ds], mod=None).keys()

efSList = [100, 110, 130, 135, 140, 145, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 1000]
#efSList = [300]
#efSList = [2000, 3000, 4000, 5000]

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
                rows = fut.result()
                for row in rows:
                    if row is None:
                        continue
                    append_row(out_csv, row)
                    results.append(row)
            except Exception as e:
                print(f"[FAILED] {ds}: {e}")

        if results:
            print(f"Saved {len(results)} rows to {out_csv}")
        else:
            print("No results.")

if __name__ == "__main__":
    main()
