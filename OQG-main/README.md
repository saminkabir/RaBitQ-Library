# Optimized Quantized Graph (OQG)

# Install
First, install [Anaconda](https://www.anaconda.com/download).  
Then start to install oqglib:  
```
sudo apt install gcc-14
git clone https://github.com/TheDatumOrg/OQG.git
conda create -n oqg -c conda-forge python=3.10
conda activate oqg
conda install -c conda-forge openblas  
pip install cmake==3.28.3
# conda install -c pytorch/label/nightly faiss-cpu  
conda install -c pytorch -c conda-forge faiss-cpu=1.13.2       
pip install pybind11==3.0.1
pip install tqdm==4.67.1

cd python_bindings
pip install --no-build-isolation .
```

# Benchmarks
First, please check the **taskset command** in example/train_test.sh, and adjust it with the number of available CPU cores you have.  
```
cd ../example
bash train_test.sh
```
Now you can check the result at train.csv and test_k100.csv.  
Please check example/train.py and example/test.py for details.  

# Set your own datasets
```
cd dataset
mkdir $YOUR_DATASET_NAME
cd $YOUR_DATASET_NAME
```
Then upload your base.fvecs, groundtruth.ivecs, query.fvecs into $YOUR_DATASET_NAME.  
Then open example/dataset/dataset_config.py and config there.

# Search Performance
![Search Performance](Average.png)

