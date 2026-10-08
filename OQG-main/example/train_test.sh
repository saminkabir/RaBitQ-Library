
# ALL_DS=(
#   uqv sald1m space1V LLAMA imageNet bigann netflix CCNEWS deep1m lendb
#   cifar nuswide ARXIV IQUIQUE astro1m audio MNIST geofon ukbench sun
#   NEIC millionSong seismic1m AGNEWS YAHOO CELEBA glove LANDMARK GOOGLEQA
#   texttoimage OBST2024 sift notre tiny5m crawl instancegm CODESEARCHNET gist
# )

ALL_DS=(gist glove sift)


# for ds in "${ALL_DS[@]}"; do
#   echo "Running dataset: $ds"
#   taskset -c 0-127 python train.py --ds "$ds"
#   python test.py --ds "$ds"
# done

# for ds in "${ALL_DS[@]}"; do
#   echo "Running dataset: $ds"
#   taskset -c 0-127 python train_mn.py --ds "$ds"
#   python test_mn.py --ds "$ds"
# done


# ALL_DS=(
#   space1V
# )


for ds in "${ALL_DS[@]}"; do
  echo "Running dataset: $ds"

  while true; do
    taskset -c 0-127 python train_mn.py --ds "$ds"
    status=$?

    if [ $status -eq 0 ]; then
      echo "Train succeeded for $ds"
      break
    else
      echo "Train failed (exit code $status), retrying..."
      sleep 1
    fi
  done

  python test_mn.py --ds "$ds"
done