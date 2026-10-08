#!/usr/bin/env python3
# Large-subspace sensitivity for OQG (reviewer comment):
#   evaluate increasingly large numbers of subspaces (128, 256, 512, 1024)
#   on GIST, to characterize the practical limit of 16-bit accumulation.
#
# Training side (mirrors fixed_bit_ablation/train_fixed_bit.py):
# for each M in M_LIST, train PQ/OPQ with M subspaces (8 bits each),
# build the graph index under the same graph settings, and save it.
#
# Usage: python train_large_m.py --ds gist
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
from dataset_config import pqopq

# Skip Ms whose index already exists (crash-resume friendly;
# train_test.sh deletes the artifacts after each dataset anyway).
SKIP_EXISTING = True
EDGE_NUM = 16   # filename tag; keep consistent with compiled numLevel0Edges

#M_LIST = [128, 256, 320, 1024]   # numbers of subspaces (8 bits per subspace)

M_LIST = [768]

CONFIG = {
    "m": 64,                 # same graph setting as train_mn.py
    "ef_construction": 600,

    "pq_dir":  "/home/cc/dataset/cb/large_subspace",       # dir for PQ codebooks
    "index_dir": "/home/cc/dataset/index/GG/large_subspace",  # dir for graph indexes

    "out_csv": os.path.join(SCRIPT_DIR, "train_large_m.csv"),
}


def paths_for(cfg: dict, dataset: str, num_subspaces: int, preprocess: str):
    pq_path = f"{cfg['pq_dir']}/{dataset}_sub{num_subspaces}_{preprocess}.npz"
    index_path = (f"{cfg['index_dir']}/{dataset}_sub{num_subspaces}_{cfg['m']}"
                  f"_{cfg['ef_construction']}_{EDGE_NUM}.mnggindex")
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

    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"

    base0 = read_vecs(conf["base"]).astype(np.float32)
    N, dim0 = base0.shape
    print(f"[{dataset}] base shape: {base0.shape}")

    num_cores = len(os.sched_getaffinity(0))
    os.makedirs(cfg["pq_dir"], exist_ok=True)
    os.makedirs(cfg["index_dir"], exist_ok=True)

    rows = []
    for num_subspaces in M_LIST:
        pq_path, index_path = paths_for(cfg, dataset, num_subspaces, preprocess)

        if SKIP_EXISTING and os.path.exists(index_path):
            print(f"[SKIP] {dataset} M={num_subspaces}: index exists at {index_path}")
            continue

        base = pad_to_mod(base0, num_subspaces)
        base = np.asarray(base, dtype=np.float32, order='C')
        dim = base.shape[1]

        t0 = time.perf_counter()
        if not os.path.exists(pq_path):
            trainPQ(base, None, pq_path, use_opq, num_subspaces)
        pq_time = time.perf_counter() - t0

        npz = np.load(pq_path)
        pq_codes = npz["codes"]
        pq_centroids = npz["centroids"]

        id_mapping = np.arange(N, dtype=np.int32)

        t0 = time.perf_counter()
        try:
            p = oqglib.MNGGIndex(m, efC, num_subspaces, dim)
        except ValueError as e:
            print(f"[UNSUPPORTED] {dataset} M={num_subspaces}: (numSubspaces={num_subspaces}, dim={dim}) "
                  f"-> add Impl<{num_subspaces},{dim}> to AnyImpl in mn_bindings.hpp and rebuild. ({e})")
            del base
            continue

        max_level_ele_ct = p.addPoints(pq_centroids, pq_codes, N, base, id_mapping)
        train_time = time.perf_counter() - t0

        p.save(index_path)
        del p, base, npz, pq_codes, pq_centroids

        rows.append({
            "dataset": dataset,
            "m": m,
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
        })
        # flush immediately: a crash on a later M must not lose this row
        append_row(cfg["out_csv"], rows[-1])
        print(f"[DONE] {dataset} M={num_subspaces} dim={dim} "
              f"train={rows[-1]['TrainSec']:.2f}s")

    del base0
    return rows


def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


parser = argparse.ArgumentParser(description="OQG large-subspace sensitivity training")
parser.add_argument("--ds", type=str, default="gist", help="ds (default: gist)")
args = parser.parse_args()

confs = get_datasets_config([args.ds], mod=None)


def main():
    results = []
    for ds, conf in confs.items():
        try:
            cfg = deepcopy(CONFIG)
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
