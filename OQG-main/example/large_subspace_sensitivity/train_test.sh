# Large-subspace sensitivity driver (mirrors fixed_bit_ablation/train_test.sh):
# GIST with M in {128, 256, 512, 1024} subspaces; retry-train -> test ->
# delete artifacts to save SSD space.

ALL_DS=(gist)

# must match CONFIG in train_large_m.py / test_large_m.py
PQ_DIR="/home/cc/dataset/cb/large_subspace"
INDEX_DIR="/home/cc/dataset/index/GG/large_subspace"

cd "$(dirname "$0")"

for ds in "${ALL_DS[@]}"; do
  echo "Running dataset: $ds"

  while true; do
    taskset -c 0-127 python train_large_m.py --ds "$ds"
    status=$?

    if [ $status -eq 0 ]; then
      echo "Train succeeded for $ds"
      break
    else
      echo "Train failed (exit code $status), retrying..."
      sleep 1
    fi
  done

  python test_large_m.py --ds "$ds"

  # free SSD space: remove this dataset's artifacts (indexes + PQ codebooks)
  rm -f "${INDEX_DIR}/${ds}_sub"*.mnggindex "${PQ_DIR}/${ds}_sub"*.npz
  echo "Cleaned artifacts for $ds"
done
