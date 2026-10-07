import pyvsag
import numpy as np
import json
from data_extractor import *
import argparse
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def calculate_recall_at(ground_truth, I, k1, k2):
    return ps.k_recall_at(ground_truth, I, k1, k2)

def hgraph_example(m, dataset_name, ef_search = 8, ef_construction = 64):
    data,query,gts=get_data_common(dataset_name)
    dim = data.shape[1]
    num_elements = data.shape[0]
    ids = range(num_elements)
    k=gts.shape[1]
    I = np.zeros((query.shape[0],k))
    # Declaring index
    index_params = json.dumps(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": dim,
            "index_param": {
                "base_quantization_type": "sq8_uniform",
                "max_degree": m,
                "ef_construction": ef_construction,
                "alpha": 1.2,
                "neighbor_sample_rate": 0.2, 
                "precise_quantization_type":"fp32", 
                "use_reorder":True,
                "build_thread_count":1
            }
        }
    )

    index = pyvsag.Index("hgraph", index_params)
    train_start_time = time.time();
    index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)
    train_end_time = time.time();
    search_params = json.dumps({"hgraph": {"ef_search": ef_search},"num_threads_searching": 1})
    all_results=[]
    cnt=0
    search_start_time = time.time()
    for q in query:
        result_ids, result_dists = index.knn_search(
            vector=q, k=k, parameters=search_params
        )
        I[cnt]=result_ids
        cnt=cnt+1
    search_end_time = time.time();
    recall=calculate_recall_at(gts,I,k,k)
    dict = {}
    dict['recall@'] = recall
    dict['model_name'] = 'VSAG-HGRAPH'
    dict['dataset_name'] = dataset_name
    dict['search-time'] =  search_end_time - search_start_time
    dict['training-time'] =  train_end_time - train_start_time
    dict['m'] = m
    dict['k'] = k
    dict['ef_search'] = ef_search
    dict['ef_construction'] = ef_construction
    print(dict)
    return dict
    
        
    

parser = argparse.ArgumentParser(description="Run MRPT ANN search on a dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Name of the dataset (e.g. 'imageNet', 'glove-100')")
parser.add_argument("--m", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")
parser.add_argument("--ef_search", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")
parser.add_argument("--ef_construction", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")
args = parser.parse_args()

hgraph_example(args.m,dataset_name=args.dataset,ef_search=args.ef_search,ef_construction=args.ef_construction)


