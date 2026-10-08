#!/usr/bin/env python3
# Memory breakdown study for OQG (reviewer comment O3) — measuring side.
# For each dataset and each level-0 edge configuration (CBCA default 16 vs
# traditional full-width 64), loads the index built by train_memory.py and
# calls MNGGIndex(.E64).memoryBreakdown(), which reports exact per-component
# sizes:
#   - LUT: PQ centroids + per-query LUT buffers (lut/qLUT/q8LUTr/q16LUT/q16LUTr)
#   - Quantized codes: PQ codes replicated inside edge lists (level0 + upper)
#   - Graph structure: neighbor IDs (level0 + upper levels)
#   - Auxiliary metadata: external-ID map, per-node container overhead,
#     visited marks
#   - Raw vectors: kept resident for refinement, so they are INCLUDED in the
#     total used for the percentage columns.
# One CSV row per (dataset, edge_num).
#
# Usage: python test_memory.py --ds gist
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
from dataset_config import suggested_subspaces, pqopq

# must match train_memory.py
EDGE_NUM_LIST = [16, 64]

CONFIG = {
    "m": 64,
    "efC": 600,

    "pq_dir":  "/home/cc/dataset/cb/memory_breakdown",       # must match train_memory.py
    "index_dir": "/home/cc/dataset/index/GG/memory_breakdown",

    "out_csv": os.path.join(SCRIPT_DIR, "memory_breakdown.csv"),
}

MB = 1024.0 * 1024.0


def get_vmrss_mb() -> float:
    """Current resident set size (VmRSS) of this process, in MB."""
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def index_class(edge_num: int):
    return {16: oqglib.MNGGIndex, 64: oqglib.MNGGIndexE64}[edge_num]


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


def paths_for(cfg: dict, dataset: str, num_subspaces: int, preprocess: str, edge_num: int):
    pq_path = f"{cfg['pq_dir']}/{dataset}_8x{num_subspaces}_{preprocess}.npz"
    index_path = (f"{cfg['index_dir']}/{dataset}_{cfg['m']}_8x{num_subspaces}"
                  f"_{cfg['efC']}_{edge_num}.mnggindex")
    return pq_path, index_path


def breakdown_one(dataset: str, cfg: dict, base: np.ndarray, edge_num: int,
                  num_subspaces: int, use_opq: bool, preprocess: str) -> dict:
    pq_path, index_path = paths_for(cfg, dataset, num_subspaces, preprocess, edge_num)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index not found: {index_path}")

    dim = base.shape[1]
    # OS-level cross-validation: RSS delta around index loading approximates
    # the true resident size of the index structures (raw vectors excluded,
    # since `base` is already resident before this point)
    gc.collect()
    rss_before = get_vmrss_mb()
    p = index_class(edge_num)(index_path, base, num_subspaces, dim)
    gc.collect()
    rss_after = get_vmrss_mb()
    measured_index_mb = max(0.0, rss_after - rss_before)

    bd = p.memoryBreakdown()
    del p
    gc.collect()

    graph = int(bd["graph_bytes"])
    codes = int(bd["codes_bytes"])
    lut = int(bd["lut_bytes"])
    meta = int(bd["metadata_bytes"])
    raw = int(bd["raw_vectors_bytes"])
    index_total = int(bd["index_total_bytes"])      # without raw vectors
    total = index_total + raw                       # raw vectors are resident (refinement)

    return {
        "Dataset": dataset,
        "EdgeNum": edge_num,
        "N": int(bd["N"]),
        "dim": int(bd["dim"]),
        "NumSubspaces": int(bd["numSubspaces"]),
        "opq": use_opq,
        "batchSize": int(bd["batchSize"]),
        "maxLevel": int(bd["maxLevel"]),
        "UpperEdgeBlocks": int(bd["upperEdgeBlocks"]),

        # component sizes (MB)
        "LUTMB": lut / MB,
        "CodesMB": codes / MB,
        "GraphMB": graph / MB,
        "MetadataMB": meta / MB,
        "RawVectorsMB": raw / MB,
        "IndexTotalMB": index_total / MB,
        "TotalWithRawMB": total / MB,

        # OS-level measurement (VmRSS delta while loading the index) and the
        # analytical/measured ratio for cross-validation
        "MeasuredIndexMB": measured_index_mb,
        "AnalyticalOverMeasured": (index_total / MB) / measured_index_mb if measured_index_mb > 0 else 0.0,
        "VmRSSMB": rss_after,

        # detailed splits (MB)
        "LUTCentroidsMB": int(bd["lut_centroids_bytes"]) / MB,
        "LUTQueryBuffersMB": int(bd["lut_query_buffers_bytes"]) / MB,
        "CodesLevel0MB": int(bd["codes_level0_bytes"]) / MB,
        "CodesUpperMB": int(bd["codes_upper_bytes"]) / MB,
        "GraphLevel0MB": int(bd["graph_level0_bytes"]) / MB,
        "GraphUpperMB": int(bd["graph_upper_bytes"]) / MB,

        # percentage of total resident memory (raw vectors included,
        # since they are required by the refinement stage)
        "LUTPct": 100.0 * lut / total,
        "CodesPct": 100.0 * codes / total,
        "GraphPct": 100.0 * graph / total,
        "MetadataPct": 100.0 * meta / total,
        "RawVectorsPct": 100.0 * raw / total,
    }


def breakdown_one_dataset(dataset: str, cfg: dict) -> list[dict]:
    all_conf = get_datasets_config([dataset], None)[dataset]

    num_subspaces = suggested_subspaces[dataset]
    use_opq = (pqopq[dataset] == "opq")
    preprocess = "opq" if use_opq else "pq"

    # load base as raw vectors so the refinement component is measured too
    base = read_vecs(all_conf["base"]).astype(np.float32)
    base = pad_to_mod(base, num_subspaces)
    base = np.asarray(base, dtype=np.float32, order='C')

    rows = []
    for edge_num in EDGE_NUM_LIST:
        try:
            row = breakdown_one(dataset, cfg, base, edge_num,
                                num_subspaces, use_opq, preprocess)
        except Exception as e:
            print(f"[FAILED] {dataset} edge{edge_num}: {e}")
            continue
        # flush immediately so a crash on a later config does not lose rows
        append_row(cfg["out_csv"], row)
        rows.append(row)
        print(f"[DONE] {dataset} edge{edge_num} | total={row['TotalWithRawMB']:.1f}MB "
              f"(LUT {row['LUTPct']:.1f}% | codes {row['CodesPct']:.1f}% | "
              f"graph {row['GraphPct']:.1f}% | meta {row['MetadataPct']:.1f}% | "
              f"raw {row['RawVectorsPct']:.1f}%) | "
              f"index analytical={row['IndexTotalMB']:.1f}MB "
              f"vs measured(RSS)={row['MeasuredIndexMB']:.1f}MB")

    del base
    gc.collect()
    return rows


def append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


parser = argparse.ArgumentParser(description="OQG memory breakdown - measuring")
parser.add_argument("--ds", type=str, required=True, help="ds")
args = parser.parse_args()

DATASETS = get_datasets_config([args.ds], mod=None).keys()


def main():
    results = []
    for ds in DATASETS:
        try:
            results.extend(breakdown_one_dataset(ds, CONFIG))
        except Exception as e:
            print(f"[FAILED] {ds}: {e}")

    if results:
        print(f"Saved {len(results)} rows to {CONFIG['out_csv']}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
