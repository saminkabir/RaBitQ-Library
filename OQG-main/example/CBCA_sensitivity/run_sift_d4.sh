#!/bin/bash
# CBCA d = D/4 sensitivity on SIFT:
#   (D=128, d=32) and (D=256, d=64); (D=64, d=16) already exists as
#   train/test_batch64_edge16.csv from the previous fixed-d=16 study.
#
# Requires oqglib compiled with MNGGIndexB128E32 / MNGGIndexB256E64
# (cd python_bindings && pip install .)
set -e
cd "$(dirname "$0")"

DS=sift

# for cfg in "128 32" "256 64"; do
for cfg in "256 64"; do
    set -- $cfg
    BATCH=$1
    EDGE=$2

    echo "=== [$DS] train batch=$BATCH edge=$EDGE ==="
    taskset -c 0-127 python train_cbca.py --ds $DS --batch $BATCH --edge $EDGE

    echo "=== [$DS] test batch=$BATCH edge=$EDGE ==="
    python test_cbca.py --ds $DS --batch $BATCH --edge $EDGE
done

echo "All done. CSVs: test_batch128_edge32.csv, test_batch256_edge64.csv"
