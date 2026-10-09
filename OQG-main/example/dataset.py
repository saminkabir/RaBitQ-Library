import numpy as np
from dataset_config import datasets
import os
from tqdm import tqdm
import struct


#base_path="./dataset/"
base_path = "/home/cc/datasets/datasets/"
#base_path = "/home/cc/OQG/example/dataset/"

def get_datasets_list():
    return datasets.keys()


def get_small_datasets(max_nbase=100000):
    small_datasets = []
    for dataset in datasets.keys():
        if datasets[dataset]["base_size"] <= max_nbase:
            small_datasets.append(dataset)

    return small_datasets

ALL_DS = ['uqv','sald1m', 'space1V', 'LLAMA', 'imageNet', 'bigann', 'netflix', 'CCNEWS', 'deep1m', 'lendb', 'cifar', 'nuswide', 'ARXIV', 'IQUIQUE', 'astro1m', 'audio', 'MNIST', 'geofon', 'ukbench', 'sun', 'NEIC', 'millionSong', 'seismic1m', 'AGNEWS', 'YAHOO', 'CELEBA', 'glove', 'LANDMARK', 'GOOGLEQA', 'texttoimage', 'OBST2024', 'sift', 'notre', 'tiny5m', 'crawl', 'instancegm', 'CODESEARCHNET', 'gist']

# mod=None means not consider padding
def get_datasets_config(datasets_name=ALL_DS, mod=None):
    if datasets_name is None:
        datasets_name = datasets.keys()

    confs = {}
    for name in datasets_name:

        if mod is None or datasets[name]['dim'] % mod == 0:
            confs[name] = {
                "path": f"{base_path}{datasets[name]['path']}",
                "base": f"{base_path}{datasets[name]['path']}{datasets[name]['base']}",
                "query": f"{base_path}{datasets[name]['path']}{datasets[name]['query']}",
                "gt": f"{base_path}{datasets[name]['path']}{datasets[name]['gt']}",
                "gt_top1000": f"{base_path}{datasets[name]['path']}gt_top1000.ivecs",
                "base_size": datasets[name]['base_size'],
                "query_size": datasets[name]['query_size'],
                "dim": datasets[name]['dim']
            }
        else:
            new_dim = int((datasets[name]['dim'] // mod + 1) * mod)
            confs[name] = {
                "path": f"{base_path}{datasets[name]['path']}",
                "base": f"{base_path}{datasets[name]['path']}base_padded{mod}.fvecs",
                "query": f"{base_path}{datasets[name]['path']}query_padded{mod}.fvecs",
                "gt": f"{base_path}{datasets[name]['path']}{datasets[name]['gt']}",
                "gt_top1000": f"{base_path}{datasets[name]['path']}gt_top1000.ivecs",
                "base_size": datasets[name]['base_size'],
                "query_size": datasets[name]['query_size'],
                "dim": new_dim
            }

        confs[name]['base_tsv'] = confs[name]['base'].rsplit('.', 1)[0] + ".tsv"
        confs[name]['query_tsv'] = confs[name]['query'].rsplit('.', 1)[0] + ".tsv"

    
    return confs



def read_ivecs(fname):
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    d = int(a[0])
    rec = d + 1


    if a.size % rec != 0:
        raise ValueError(f"Corrupt ivecs? a.size={a.size} not divisible by (d+1)={rec}")

    return a.reshape(-1, rec)[:, 1:]  


def read_fvecs(fname):
    x = read_ivecs(fname).view(np.float32)
    return x

def read_vecs(fname):
    suffix = fname.split(".")[-1]
    if suffix == "fvecs":
        return read_fvecs(fname)
    elif suffix == "bvecs":
        return read_bvecs(fname)
    elif suffix == "ivecs":
        return read_ivecs(fname)
    else:
        assert(True, f"{fname} is not supported!")

def read_bvecs(fname):
    raw = np.fromfile(fname, dtype=np.uint8)    
    d = int(np.frombuffer(raw[:4].tobytes(), dtype=np.int32)[0]) 
    rec_size = 4 + d
    n = raw.size // rec_size                
    data = raw.reshape(n, rec_size)[:, 4:]    
    return data.copy()                     

def write_ivecs(fname, a):
    n, d = a.shape
    b = np.empty((n, d+1), dtype=np.int32)
    b[:, 0] = d
    b[:, 1:] = a
    b.tofile(fname)


def write_fvecs(fname, a):
    n, d = a.shape
    dt = np.dtype([('dim', np.int32), ('vec', np.float32, d)])
    b = np.empty(n, dtype=dt)
    b['dim'] = d
    b['vec'] = a
    b.tofile(fname)


def write_bvecs(fname, a: np.ndarray):
    if a.ndim != 2:
        raise ValueError("Input must be 2D array")
    if a.dtype != np.uint8:
        raise TypeError(f"Input must be uint8, got {a.dtype}")
    n, d = a.shape

    dt = np.dtype([('dim', np.int32), ('vec', np.uint8, d)])
    b = np.empty(n, dtype=dt)
    b['dim'] = d
    b['vec'] = a
    b.tofile(fname)


def check_path(datasets_name=None):
    confs = get_datasets_config(datasets_name, None)
    for name, conf in confs.items():
        if(not os.path.exists(conf['base'])):
            print(f"\n{conf['base']} dose not exist")
        if(not os.path.exists(conf['query'])):
            print(f"\n{conf['query']} dose not exist")
        if(not os.path.exists(conf['gt'])):
            print(f"\n{conf['gt']} dose not exist")
    print("Check Path Done")


def check_shapes(datasets_name=None, padding_mod=None):
    confs = get_datasets_config(datasets_name, padding_mod)
    topk=100
    unsual = []
    for name, conf in confs.items():
        print(f"Checking {name}")
        base = read_vecs(conf['base'])
        query = read_vecs(conf['query'])
        gt = read_ivecs(conf['gt'])
        assert(base.shape[1] == query.shape[1])
        assert base.shape[1] == conf['dim'], f"{base.shape[1]} {conf['dim']}"
        assert base.shape[0] == conf['base_size'], f"{base.shape[0]} {conf['base_size']}"
        assert query.shape[0] == conf['query_size'], f"{query.shape[0]} {conf['query_size']}"
        assert(query.shape[1] == conf['dim'])
        assert(gt.shape[0] == conf['query_size'])
        if(gt.shape[1] != topk):
            unsual.append(((name, gt.shape[1])))
    print(unsual)


def padding_datasets(datasets_name=None, mod=16):
    confs = get_datasets_config(datasets_name, None)
    loop = tqdm(confs.items())
    for name, conf in loop:
        loop.set_description_str(f"{name}|mod: {mod}")
        if conf['dim'] % mod != 0:
            new_dim = int((conf['dim'] // mod + 1) * mod)
        else:
            print(f"{name}: Already aligned to {mod}, skipping.")
            continue

        if name not in ["sift10M"]:
            old_base = read_fvecs(conf['base'])
            old_query = read_fvecs(conf['query'])
        else:
            old_base = read_bvecs(conf['base'])
            old_query = read_bvecs(conf['query'])

        def pad_data(data, old_dim, new_dim):
            pad_width = new_dim - old_dim
            if name not in ["sift10M"]:
                noise = np.random.normal(loc=0.0, scale=1e-20, size=(data.shape[0], pad_width)).astype(np.float32)
            else:
                noise = np.random.normal(loc=0.0, scale=1e-20, size=(data.shape[0], pad_width)).astype(np.uint8)
            return np.hstack([data, noise])

        base_padded = pad_data(old_base, conf['dim'], new_dim)
        query_padded = pad_data(old_query, conf['dim'], new_dim)

        base_out_path = f"{conf['path']}base_padded{mod}.fvecs"
        query_out_path = f"{conf['path']}query_padded{mod}.fvecs"

        if name not in ["sift10M"]:
            write_fvecs(base_out_path, base_padded)
            write_fvecs(query_out_path, query_padded)
        else:
            write_bvecs(base_out_path, base_padded)
            write_bvecs(query_out_path, query_padded)            

        loop.write(f"{name}: Padded and saved to {base_out_path} and {query_out_path}. {old_base.shape[1]} -> {base_padded.shape[1]}")


def get_human_readable_size(path: str) -> str:
    """Return the file size in human readable format (KB, MB, GB, ...)."""
    size_bytes = os.path.getsize(path)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {units[i]}"

import os

def latex_table_generate_body(datasets_name, fname):
    confs = get_datasets_config(datasets_name, None)

    # Sort alphabetically for stable table order
    items = sorted(confs.items(), key=lambda x: x[0].lower())

    # Helper to format bytes → MB/GB
    def format_size(size_bytes):
        if size_bytes >= 1024**3:
            return f"{size_bytes / (1024**3):.2f}GB"
        if size_bytes >= 1024**2:
            return f"{size_bytes / (1024**2):.0f}MB"
        if size_bytes >= 1024:
            return f"{size_bytes / 1024:.0f}KB"
        return f"{size_bytes}B"

    latex = ""
    row = []

    for name, conf in items:

        # --- Read base file size ---
        base_path = conf["base"]      # a file path
        size_bytes = os.path.getsize(base_path)
        size_str = format_size(size_bytes)

        # --- Format one dataset entry ---
        entry = (
            f"{name.upper()}~\\cite{{}} & "
            f"{conf['dim']} & {conf['base_size']} & {conf['query_size']} & {size_str}"
        )

        row.append(entry)

        # Every 2 datasets → 1 row
        if len(row) == 2:
            latex += row[0] + " & " + row[1] + " \\\\\n"
            row = []

    # If #datasets is odd → last row only one dataset
    if len(row) == 1:
        latex += row[0] + " & & & & \\\\\n"

    # Output to file
    with open(fname, "w+") as f:
        f.write(latex)




def get_dataset_stat(datasets_name=None):
    confs = get_datasets_config(datasets_name, None)

    # Sort items by base_size (largest first)
    sorted_confs = sorted(confs.items(), key=lambda kv: kv[1]['base_size']*kv[1]['dim'], reverse=True)

    loop = tqdm(confs.items())
    for name, conf in loop:
        dim = conf['dim']
        d_size = conf['base_size']
        print(f"{name}: {dim}x{d_size}={get_human_readable_size(conf['base'])}, {4*dim*d_size/(1024**3)}")


import tarfile
def pack_datasets(out_path="datasets.tar.gz"):
    """
    将 get_datasets_config() 返回的所有数据集打包为 tar.gz。
    解压后目录结构：
        dataset/<name>/<原始文件名>
    其中 <name> 取自 conf['path'] 的最后一段目录名。
    """
    confs = get_datasets_config(mod=None)
    with tarfile.open(out_path, "w:gz") as tar:
        loop = tqdm(confs.items())
        for ds_name, conf in loop:
            loop.set_description_str(f"{ds_name}")
            ds_dirname = os.path.basename(str(conf["path"]).rstrip("/"))
            base_arcdir = os.path.join("dataset", ds_dirname)

            for key in ("base", "query", "gt"):
                src = conf.get(key)
                if not src:
                    print(f"[skip] missing key '{key}' for {ds_name}")
                    continue
                if not os.path.exists(src):
                    print(f"[skip] missing file {src}")
                    continue
                fname = os.path.basename(src)  # 使用原始文件名
                tar.add(src, arcname=os.path.join(base_arcdir, fname))

    print(f"✅ 打包完成: {out_path}")

