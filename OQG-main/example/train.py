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
import faiss

RETRAIN = True

def read_ivecs(fname: str) -> np.ndarray:
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def read_fvecs(fname: str) -> np.ndarray:
    return read_ivecs(fname).view("float32")

def pad_to_mod(arr: np.ndarray, mod: int, pad_value: float = 0.0) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("只支持二维数组输入")
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
    elif num_bits == 6:
        assert(False)
        # packed_codes = faiss.vector_to_array(index_flat.codes)
        # codes = unpack_6bit_codes(packed_codes, ntotal=index_flat.ntotal, num_subspaces=num_subspaces)
        # return codes


def get_pq_centroids(index_flat):
    cen = faiss.vector_to_array(index_flat.pq.centroids)
    return cen.reshape(index_flat.pq.M, index_flat.pq.ksub, index_flat.pq.dsub)


import numpy as np
import faiss

# -------------------------
# Utilities
# -------------------------

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


# -------------------------
# Main training function
# -------------------------

def trainPQ(
    base,
    index_path,
    pq_path,
    use_opq: bool,
    num_subspaces: int,
    *,
    num_bits: int = 8,
    max_train: int = 5_000_000,
    seed: int = 123,
    verbose: bool = True,
):
    """
    Train FAISS PQ or OPQ+PQ, add full base vectors, and export PQ centroids/codes (+ OPQ matrix).

    Parameters
    ----------
    base : np.ndarray
        Base vectors, shape (N, d). Can be memmap.
    index_path : str or None
        Path to save FAISS index (optional).
    pq_path : str
        Path to save exported pq metadata (.npz).
    use_opq : bool
        Whether to use OPQ rotation before PQ.
    num_subspaces : int
        M in PQ/OPQ.
    num_bits : int
        nbits in PQ (default 8).
    max_train : int
        Max number of vectors used for training OPQ/PQ (default 5,000,000).
    seed : int
        Random seed for sampling.
    verbose : bool
        Print progress messages.
    """

    # Make sure base is float32 contiguous for FAISS
    #base = ensure_faiss_float32(base)
    dim = base.shape[1]

    # Training set selection (apply to BOTH OPQ and PQ)
    train_base = choose_train_base(base, max_train=max_train, seed=seed)
    train_base = ensure_faiss_float32(train_base)

    if verbose:
        print(f"[trainPQ] dim={dim}, N={base.shape[0]}, trainN={train_base.shape[0]}, "
              f"M={num_subspaces}, nbits={num_bits}, use_opq={use_opq}")

    # -------------------------
    # Index Training / Building
    # -------------------------
    if use_opq:
        if verbose:
            print("Step 1: Train OPQ rotation (on train_base)")
        opq = faiss.OPQMatrix(dim, num_subspaces)
        # Optional knobs:
        # opq.niter = 30
        # opq.niter_pq = 10
        # opq.niter_pq_0 = 50
        opq.train(train_base)

        if verbose:
            print("Step 2: Build IndexPreTransform(OPQ -> PQ)")
        inner_pq = faiss.IndexPQ(dim, num_subspaces, num_bits)
        pq_index = faiss.IndexPreTransform(opq, inner_pq)  # OPQ + PQ chain

        if verbose:
            print("Step 3: Train PQ (on train_base)")
        pq_index.train(train_base)

        if verbose:
            print("Step 4: Add / Encode full base")
        pq_index.add(base)

        if verbose:
            print("Step 5: Save FAISS index")
        if index_path is not None:
            faiss.write_index(pq_index, index_path)

    else:
        if verbose:
            print("Step 1: Train PQ (on train_base)")
        pq_index = faiss.IndexPQ(dim, num_subspaces, num_bits)
        pq_index.train(train_base)

        if verbose:
            print("Step 2: Add / Encode full base")
        pq_index.add(base)

        if verbose:
            print("Step 3: Save FAISS index")
        if index_path is not None:
            faiss.write_index(pq_index, index_path)

    # -------------------------
    # Export PQ codes and centroids
    # -------------------------
    if verbose:
        print("Step 6: Export PQ meta-data (centroids/codes" + ("/opq_matrix" if use_opq else "") + ")")

    if use_opq:
        # Extract inner PQ + OPQ transform from the chain
        inner = faiss.downcast_index(pq_index.index)                 # IndexPQ
        vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))    # OPQMatrix (VectorTransform)

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

    if verbose:
        print("Done.")

    return num_subspaces


def train_one_dataset(dataset: str, conf: dict, cfg: dict) -> dict:

    m               = cfg["m"]
    num_bits        = 8
    efC             = cfg["ef_construction"]
    pq_dir        = cfg["pq_dir"]
    index_dir       = cfg["index_dir"]

    base_path  = conf["base"]

    num_subspaces = suggested_subspaces[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"


    base  = read_vecs(base_path).astype(np.float32)
    base  = pad_to_mod(base, num_subspaces)
    base = np.asarray(base, dtype=np.float32, order='C')
    N, dim = base.shape


    pq_path        = f"{pq_dir}/{dataset}_8x{num_subspaces}_{preprocess}.npz"
    index_path     = f"{index_dir}/{dataset}_{m}_8x{num_subspaces}_{efC}.ggindex"
    #pq_index_path = f"{pq_dir}/{dataset}_8x{num_subspaces}_{preprocess}.pqindex"
    pq_index_path = None

    if not RETRAIN and not os.path.exists(index_path):
        return None

    t0 = time.perf_counter()
    if not RETRAIN and not os.path.exist(pq_index_path):
        trainPQ(base, pq_index_path, pq_path, use_opq, num_subspaces)
    pq_time = time.perf_counter() - t0
    t0 = time.perf_counter()

    if not RETRAIN and not os.path.exists(pq_path):
        raise FileNotFoundError(f"PQ Files not Found: {pq_path}")
    npz = np.load(pq_path)
    if use_opq:
        pq_codes      = npz["codes"]
        pq_centroids  = npz["centroids"]
    else:
        pq_codes      = npz["codes"]
        pq_centroids  = npz["centroids"]

    os.makedirs(index_dir, exist_ok=True)
    id_mapping = np.arange(N, dtype=np.int32)

    t0 = time.perf_counter()
    p = oqglib.GGIndex(m, efC, num_subspaces, dim)

    max_level_ele_ct = p.addPoints(pq_centroids, pq_codes, N, base, id_mapping)
    train_time = time.perf_counter() - t0

    p.save(index_path)
    del base
    del p

    num_cores =len(os.sched_getaffinity(0))


    return {
        "dataset": dataset,
        "m": m,
        "num_bits": num_bits,
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

def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


CONFIG = {
    "m": 64,
    "ef_construction": 600,

    "pq_dir":  "./index/cb", # dir path for PQ codebook
    "index_dir": "./index/GG", # dir path for graph index

    "out_csv": "train.csv",

    "max_workers": 1,
}

os.makedirs(CONFIG['pq_dir'], exist_ok=True)
os.makedirs(CONFIG['index_dir'], exist_ok=True)

import argparse

parser = argparse.ArgumentParser(description="Demo for reading a parameter")
parser.add_argument("--ds", type=str, required=True, help="ds")
args = parser.parse_args()

confs = get_datasets_config([args.ds], mod=None)

def main():
    out_csv = CONFIG["out_csv"]

    results = []

    for ds, conf in confs.items():
        try:
            cfg = deepcopy(CONFIG)
            row = train_one_dataset(ds, conf, cfg)
        except Exception as e:
            print(f"[FAILED] {ds}: {e}")
        if row is None:
            continue
        append_row(out_csv, row)
        results.append(row)
        print(f"[DONE] {ds} | N={row['N']} dim={row['dim']} "
            f"subspaces={row['num_subspaces']} opq={row['opq']} "
            f"train_time={row['TrainSec']:.2f}s")

    if results:
        print(f"Saved {len(results)} rows to {out_csv}")
    else:
        print("No results.")

if __name__ == "__main__":
    main()

