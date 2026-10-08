#!/usr/bin/env python3
# Memory breakdown study for OQG (reviewer comment O3):
#   report memory consumption across lookup tables, quantized codes,
#   graph structures, and auxiliary metadata.
#
# Training side (mirrors train_mn.py with the default configuration:
# m=64, ef_construction=600, per-dataset suggested subspaces, pq/opq per
# dataset_config). For each dataset, builds and saves one index per
# level-0 edge configuration in EDGE_NUM_LIST: the CBCA default (16) and
# the traditional full-width edge list (64), so the comparison shows
# where the memory savings come from.
#
# Usage: python train_memory.py --ds gist
import os
import sys
import csv
import time
import argparse
import numpy as np
import faiss
import oqglib
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OQG_EXAMPLE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, OQG_EXAMPLE_DIR)
from dataset import get_datasets_config, read_vecs
from dataset_config import suggested_subspaces, pqopq

# Skip configs whose index already exists (crash-resume friendly;
# train_test.sh deletes the artifacts after each dataset anyway).
SKIP_EXISTING = True

# CBCA default (16) vs traditional full-width edge list (64)
EDGE_NUM_LIST = [16, 64]

CONFIG = {
    "m": 64,                 # default graph setting (same as train_mn.py)
    "ef_construction": 600,

    "pq_dir":  "/home/cc/dataset/cb/memory_breakdown",       # dir for PQ codebooks
    "index_dir": "/home/cc/dataset/index/GG/memory_breakdown",  # dir for graph indexes

    "out_csv": os.path.join(SCRIPT_DIR, "train_memory.csv"),
}


def index_class(edge_num: int):
    return {16: oqglib.MNGGIndex, 64: oqglib.MNGGIndexE64}[edge_num]


def paths_for(cfg: dict, dataset: str, num_subspaces: int, preprocess: str, edge_num: int):
    pq_path = f"{cfg['pq_dir']}/{dataset}_8x{num_subspaces}_{preprocess}.npz"
    index_path = (f"{cfg['index_dir']}/{dataset}_{cfg['m']}_8x{num_subspaces}"
                  f"_{cfg['ef_construction']}_{edge_num}.mnggindex")
    return pq_path, index_path


# ──────────────────────────────────────────────
# helpers copied from train_mn.py (identical behavior)
# ──────────────────────────────────────────────
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


def get_flat_codes(index_flat, num_subspaces, num_bits):
    if num_bits == 8 or num_bits == 10:
        return faiss.vector_to_array(index_flat.codes).reshape(
            index_flat.ntotal, num_subspaces)
    assert False, f"num_bits={num_bits} not supported"


def get_pq_centroids(index_flat):
    cen = faiss.vector_to_array(index_flat.pq.centroids)
    return cen.reshape(index_flat.pq.M, index_flat.pq.ksub, index_flat.pq.dsub)


def ensure_faiss_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return np.ascontiguousarray(x)


def sample_training_vectors(x, n_samples: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    n_samples = min(int(n_samples), int(N))
    if n_samples == N:
        return x
    idx = rng.choice(N, size=n_samples, replace=False)
    return x[idx]


def choose_train_base(base: np.ndarray, max_train: int, seed: int = 123) -> np.ndarray:
    if base.shape[0] > max_train:
        return sample_training_vectors(base, n_samples=max_train, seed=seed)
    return base


def trainPQ(base, index_path, pq_path, use_opq, num_subspaces, *,
            num_bits=8, max_train=5_000_000, seed=123, verbose=True):
    dim = base.shape[1]
    train_base = choose_train_base(base, max_train=max_train, seed=seed)
    train_base = ensure_faiss_float32(train_base)

    if verbose:
        print(f"[trainPQ] dim={dim}, N={base.shape[0]}, trainN={train_base.shape[0]}, "
              f"M={num_subspaces}, nbits={num_bits}, use_opq={use_opq}")

    if use_opq:
        opq = faiss.OPQMatrix(dim, num_subspaces)
        opq.train(train_base)
        inner_pq = faiss.IndexPQ(dim, num_subspaces, num_bits)
        pq_index = faiss.IndexPreTransform(opq, inner_pq)
        pq_index.train(train_base)
        pq_index.add(base)
        if index_path is not None:
            faiss.write_index(pq_index, index_path)
    else:
        pq_index = faiss.IndexPQ(dim, num_subspaces, num_bits)
        pq_index.train(train_base)
        pq_index.add(base)
        if index_path is not None:
            faiss.write_index(pq_index, index_path)

    if use_opq:
        inner = faiss.downcast_index(pq_index.index)
        vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
        opq_matrix = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        pq_for_export = inner
    else:
        opq_matrix = None
        pq_for_export = pq_index

    centroids = get_pq_centroids(pq_for_export).transpose(1, 0, 2)
    codes = get_flat_codes(pq_for_export, num_subspaces, num_bits)

    if use_opq:
        np.savez(pq_path, centroids=centroids, codes=codes, opq_matrix=opq_matrix)
    else:
        np.savez(pq_path, centroids=centroids, codes=codes)

    return num_subspaces


# ──────────────────────────────────────────────
# training
# ──────────────────────────────────────────────
def train_one_dataset(dataset: str, conf: dict, cfg: dict) -> list[dict]:
    m   = cfg["m"]
    efC = cfg["ef_construction"]

    num_subspaces = suggested_subspaces[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"

    base = read_vecs(conf["base"]).astype(np.float32)
    base = pad_to_mod(base, num_subspaces)
    base = np.asarray(base, dtype=np.float32, order='C')
    N, dim = base.shape
    print(f"[{dataset}] base shape: {base.shape}")

    num_cores = len(os.sched_getaffinity(0))
    os.makedirs(cfg["pq_dir"], exist_ok=True)
    os.makedirs(cfg["index_dir"], exist_ok=True)

    rows = []
    for edge_num in EDGE_NUM_LIST:
        pq_path, index_path = paths_for(cfg, dataset, num_subspaces, preprocess, edge_num)
        if SKIP_EXISTING and os.path.exists(index_path):
            print(f"[SKIP] {dataset} edge{edge_num}: index exists at {index_path}")
            continue

        # PQ codebook is shared between edge configs (independent of edge_num)
        t0 = time.perf_counter()
        if not os.path.exists(pq_path):
            trainPQ(base, None, pq_path, use_opq, num_subspaces)
        pq_time = time.perf_counter() - t0

        npz = np.load(pq_path)
        pq_codes = npz["codes"]
        pq_centroids = npz["centroids"]

        id_mapping = np.arange(N, dtype=np.int32)

        t0 = time.perf_counter()
        p = index_class(edge_num)(m, efC, num_subspaces, dim)
        max_level_ele_ct = p.addPoints(pq_centroids, pq_codes, N, base, id_mapping)
        train_time = time.perf_counter() - t0

        p.save(index_path)
        del p, npz, pq_codes, pq_centroids

        row = {
            "dataset": dataset,
            "m": m,
            "edge_num": edge_num,
            "num_bits": 8,
            "num_subspaces": num_subspaces,
            "opq": int(use_opq),
            "ef_construction": efC,
            "N": int(N),
            "dim": int(dim),
            "pq_time_s": float(pq_time),
            "indexing_time_s": float(train_time),
            "TrainSec": float(train_time + pq_time),
            "max_level_ele_ct": int(max_level_ele_ct),
            "num_cores": num_cores,
        }
        # flush immediately: a crash on a later config must not lose this row
        append_row(cfg["out_csv"], row)
        rows.append(row)
        print(f"[DONE] {dataset} edge{edge_num} | N={N} dim={dim} "
              f"M={num_subspaces} train={row['TrainSec']:.2f}s")

    del base
    return rows


def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


parser = argparse.ArgumentParser(description="OQG memory breakdown - training")
parser.add_argument("--ds", type=str, required=True, help="ds")
args = parser.parse_args()

confs = get_datasets_config([args.ds], mod=None)


def main():
    results = []
    for ds, conf in confs.items():
        try:
            cfg = deepcopy(CONFIG)
            # rows are appended to the CSV inside train_one_dataset (per edge_num)
            results.extend(train_one_dataset(ds, conf, cfg))
        except Exception as e:
            print(f"[FAILED] {ds}: {e}")
            sys.exit(1)   # let train_test.sh retry

    if results:
        print(f"Saved {len(results)} rows to {CONFIG['out_csv']}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
