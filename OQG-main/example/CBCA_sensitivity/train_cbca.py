#!/usr/bin/env python3
# CBCA d = D/4 sensitivity (training side):
#   build the graph index with degree D (= batchSize) and CBCA-compressed
#   level-0 edge count d (= numLevel0Edges), e.g. D=128 -> d=32, D=256 -> d=64.
#
# The (D, d) pair maps to a dedicated compiled binding class, see
# python_bindings/mn_bindings.hpp. PQ codebooks are reused from previous
# runs when available so all (D, d) configs share the same quantizer.
#
# Usage: python train_cbca.py --ds sift --batch 128 --edge 32
import os
import sys
import csv
import time
import shutil
import argparse
import numpy as np
import oqglib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OQG_EXAMPLE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, OQG_EXAMPLE_DIR)
from dataset import get_datasets_config, read_vecs
from dataset_config import suggested_subspaces, pqopq

EF_CONSTRUCTION = 600

# reuse an existing PQ codebook from these locations before retraining,
# so every (D, d) config is built on the exact same quantizer
SHARED_PQ_DIRS = [
    "/home/cc/dataset/cb",
    "/home/cc/dataset/cb/batch64edge16",
]


def get_index_class(batch: int, edge: int):
    legacy = {(64, 16): "MNGGIndex", (64, 64): "MNGGIndexE64"}
    name = legacy.get((batch, edge), f"MNGGIndexB{batch}E{edge}")
    cls = getattr(oqglib, name, None)
    if cls is None:
        raise RuntimeError(
            f"oqglib.{name} not found: recompile python_bindings with "
            f"(batchSize={batch}, numLevel0Edges={edge}) exposed")
    return cls


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


# ──────────────────────────────────────────────
# PQ training (same as train_mn.py, imported lazily so that a reused
# codebook does not require faiss at all)
# ──────────────────────────────────────────────
def train_pq(base, pq_path, use_opq: bool, num_subspaces: int,
             num_bits: int = 8, max_train: int = 5_000_000, seed: int = 123):
    import faiss

    def get_flat_codes(index_flat):
        return faiss.vector_to_array(index_flat.codes).reshape(
            index_flat.ntotal, num_subspaces)

    def get_pq_centroids(index_flat):
        cen = faiss.vector_to_array(index_flat.pq.centroids)
        return cen.reshape(index_flat.pq.M, index_flat.pq.ksub, index_flat.pq.dsub)

    dim = base.shape[1]

    train_base = base
    if base.shape[0] > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(base.shape[0], size=max_train, replace=False)
        train_base = base[idx]
    train_base = np.ascontiguousarray(train_base, dtype=np.float32)

    print(f"[train_pq] dim={dim}, N={base.shape[0]}, trainN={train_base.shape[0]}, "
          f"M={num_subspaces}, nbits={num_bits}, use_opq={use_opq}")

    if use_opq:
        opq = faiss.OPQMatrix(dim, num_subspaces)
        opq.train(train_base)
        inner_pq = faiss.IndexPQ(dim, num_subspaces, num_bits)
        pq_index = faiss.IndexPreTransform(opq, inner_pq)
        pq_index.train(train_base)
        pq_index.add(base)
        inner = faiss.downcast_index(pq_index.index)
        vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
        opq_matrix = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        pq_for_export = inner
    else:
        pq_index = faiss.IndexPQ(dim, num_subspaces, num_bits)
        pq_index.train(train_base)
        pq_index.add(base)
        opq_matrix = None
        pq_for_export = pq_index

    centroids = get_pq_centroids(pq_for_export).transpose(1, 0, 2)
    codes = get_flat_codes(pq_for_export)

    if use_opq:
        np.savez(pq_path, centroids=centroids, codes=codes, opq_matrix=opq_matrix)
    else:
        np.savez(pq_path, centroids=centroids, codes=codes)


def ensure_pq(pq_path: str):
    """Reuse an existing codebook if possible; return True when found/copied."""
    if os.path.exists(pq_path):
        return True
    fname = os.path.basename(pq_path)
    for d in SHARED_PQ_DIRS:
        cand = os.path.join(d, fname)
        if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath(pq_path):
            print(f"[pq] reuse codebook: {cand}")
            shutil.copyfile(cand, pq_path)
            return True
    return False


def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def train_one_dataset(dataset: str, batch: int, edge: int) -> dict:
    m = batch                      # graph degree D == batchSize
    efC = EF_CONSTRUCTION

    pq_dir = f"/home/cc/dataset/cb/batch{batch}edge{edge}"
    index_dir = f"/home/cc/dataset/index/GG/batch{batch}edge{edge}"
    os.makedirs(pq_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    conf = get_datasets_config([dataset], mod=None)[dataset]

    num_subspaces = suggested_subspaces[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"

    base = read_vecs(conf["base"]).astype(np.float32)
    base = pad_to_mod(base, num_subspaces)
    base = np.asarray(base, dtype=np.float32, order='C')
    N, dim = base.shape

    pq_path = f"{pq_dir}/{dataset}_8x{num_subspaces}_{preprocess}.npz"
    index_path = f"{index_dir}/{dataset}_{m}_8x{num_subspaces}_{efC}_{edge}.mnggindex"

    t0 = time.perf_counter()
    if not ensure_pq(pq_path):
        train_pq(base, pq_path, use_opq, num_subspaces)
    pq_time = time.perf_counter() - t0

    npz = np.load(pq_path)
    pq_codes = npz["codes"]
    pq_centroids = npz["centroids"]

    id_mapping = np.arange(N, dtype=np.int32)

    IndexCls = get_index_class(batch, edge)
    assert IndexCls.batchSize == batch and IndexCls.numLevel0Edges == edge

    t0 = time.perf_counter()
    p = IndexCls(m, efC, num_subspaces, dim)
    max_level_ele_ct = p.addPoints(pq_centroids, pq_codes, N, base, id_mapping)
    train_time = time.perf_counter() - t0

    p.save(index_path)
    del base
    del p

    num_cores = len(os.sched_getaffinity(0))

    return {
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
    }


def main():
    parser = argparse.ArgumentParser(description="CBCA sensitivity training")
    parser.add_argument("--ds", type=str, required=True, help="dataset name")
    parser.add_argument("--batch", type=int, required=True,
                        help="graph degree D (= batchSize)")
    parser.add_argument("--edge", type=int, required=True,
                        help="CBCA level-0 edge count d (= numLevel0Edges)")
    args = parser.parse_args()

    out_csv = os.path.join(SCRIPT_DIR, f"train_batch{args.batch}_edge{args.edge}.csv")

    row = train_one_dataset(args.ds, args.batch, args.edge)
    append_row(out_csv, row)
    print(f"[DONE] {args.ds} | batch={args.batch} edge={args.edge} "
          f"N={row['N']} dim={row['dim']} subspaces={row['num_subspaces']} "
          f"train_time={row['TrainSec']:.2f}s -> {out_csv}")


if __name__ == "__main__":
    main()
