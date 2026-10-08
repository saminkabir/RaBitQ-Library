# Memory breakdown driver (mirrors fixed_bit_ablation/train_test.sh):
# for each dataset: retry-train -> measure memory breakdown -> delete that
# dataset's index/codebook files to save SSD space.

# ALL_DS=(
#   uqv sald1m space1V LLAMA imageNet bigann netflix CCNEWS deep1m lendb
#   cifar nuswide ARXIV IQUIQUE astro1m audio MNIST geofon ukbench sun
#   NEIC millionSong seismic1m AGNEWS YAHOO CELEBA glove LANDMARK GOOGLEQA
#   texttoimage OBST2024 sift notre tiny5m crawl instancegm CODESEARCHNET gist
# )

ALL_DS=(
  uqv sald1m space1V LLAMA imageNet bigann netflix CCNEWS deep1m
  cifar nuswide ARXIV IQUIQUE astro1m audio MNIST ukbench sun
  NEIC millionSong seismic1m YAHOO glove LANDMARK
  texttoimage sift notre tiny5m crawl gist
)

# must match CONFIG in train_memory.py / test_memory.py
PQ_DIR="/home/cc/dataset/cb/memory_breakdown"
INDEX_DIR="/home/cc/dataset/index/GG/memory_breakdown"

cd "$(dirname "$0")"

for ds in "${ALL_DS[@]}"; do
  echo "Running dataset: $ds"

  while true; do
    taskset -c 0-127 python train_memory.py --ds "$ds"
    status=$?

    if [ $status -eq 0 ]; then
      echo "Train succeeded for $ds"
      break
    else
      echo "Train failed (exit code $status), retrying..."
      sleep 1
    fi
  done

  python test_memory.py --ds "$ds"

  # free SSD space: remove this dataset's artifacts (indexes + PQ codebooks)
  rm -f "${INDEX_DIR}/${ds}_"*.mnggindex "${PQ_DIR}/${ds}_"*.npz
  echo "Cleaned artifacts for $ds"
done
