# Fixed-bit ablation driver (mirrors OQG/example/train_test.sh):
# for each dataset: retry-train -> test -> delete that dataset's
# index/codebook files to save SSD space.

# ALL_DS=(
#   uqv sald1m space1V LLAMA imageNet bigann netflix CCNEWS deep1m lendb
#   cifar nuswide ARXIV IQUIQUE astro1m audio MNIST geofon ukbench sun
#   NEIC millionSong seismic1m AGNEWS YAHOO CELEBA glove LANDMARK GOOGLEQA
#   texttoimage OBST2024 sift notre tiny5m crawl instancegm CODESEARCHNET gist
# )

ALL_DS=(
  space1V bigann lendb nuswide IQUIQUE astro1m audio MNIST geofon ukbench sun
  NEIC millionSong seismic1m AGNEWS YAHOO CELEBA glove LANDMARK GOOGLEQA
  texttoimage OBST2024 sift notre tiny5m crawl instancegm CODESEARCHNET gist
)

# must match CONFIG in train_fixed_bit.py / test_fixed_bit.py
PQ_DIR="/home/cc/dataset/cb/fixed_bit"
INDEX_DIR="/home/cc/dataset/index/GG/fixed_bit"

cd "$(dirname "$0")"

for ds in "${ALL_DS[@]}"; do
  echo "Running dataset: $ds"

  while true; do
    taskset -c 0-127 python train_fixed_bit.py --ds "$ds"
    status=$?

    if [ $status -eq 0 ]; then
      echo "Train succeeded for $ds"
      break
    else
      echo "Train failed (exit code $status), retrying..."
      sleep 1
    fi
  done

  python test_fixed_bit.py --ds "$ds"

  # free SSD space: remove this dataset's artifacts (indexes + PQ codebooks)
  rm -f "${INDEX_DIR}/${ds}_b"*.mnggindex "${PQ_DIR}/${ds}_b"*.npz
  echo "Cleaned artifacts for $ds"
done
