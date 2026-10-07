import sys
import os

x = [
    'yahoomusic', 'deep', 'landmark-dino-768-cosine', 'instancegm', 'txed',
    'millionSong', 'Iquique', 'movielens', 'nuswide', 'audio', 'vcseis', 'OBS',
    'text-to-image', 'yi-128-ip', 'glove', 'sift', 'landmark-nomic-768-normalized',
    'uqv', 'simplewiki-openai-3072-normalized', 'MNIST', 'geofon',
    'arxiv-nomic-768-normalized', 'ethz', 'lendb', 'enron', 'random', 'cifar',
    'Meier2019JGR', 'space1V', 'yandex-200-cosine', 'coco-nomic-768-normalized',
    'word2vec', 'imagenet-clip-512-normalized', 'ukbench',
    'laion-clip-512-normalized', 'gist', 'imagenet-align-640-normalized',
    'nytimes', 'seismic1m', 'NEIC', 'netflix', 'ISC_EHB_DepthPhases', 'imageNet',
    'gooaq-distilroberta-768-normalized', 'celeba-resnet-2048-cosine', 'Music',
    'llama-128-ip', 'crawl', 'trevi', 'PNW', 'notre',
    'agnews-mxbai-1024-euclidean', 'OBST2024', 'bigann', 'sun', 'stead',
    'sald1m', 'yahoo-minilm-384-normalized'
]

excludes = []

tmp = '/usr/bin/time -v python vsag_hgraph.py --dataset [dataset] --m [m] --ef_search [ef_search] --ef_construction [ef_construction] &> logVSAGhgraph/[dataset]-[m]-[ef_search]-[ef_construction]-1.2.txt'

excluded_cpus = {9, 12, 13, 16, 31, 33, 35, 51, 52, 55, 70, 71, 74}
all_cpus = list(range(0,159))
allowed_cpus = [c for c in all_cpus if c not in excluded_cpus]

commands = []

for th in [(16,16,16),(16,8,16),(24,12,24),(32,16,32),(48,24,48),(64,32,64),(96,96,96),(128,128,128)]:
    for dataset_info in x:
        dataset = dataset_info
        if dataset not in excludes:
            text = tmp + ''
            text = text.replace('[dataset]', dataset)
            text = text.replace('[m]', str(th[0]))
            text = text.replace('[ef_search]', str(th[1]))
            text = text.replace('[ef_construction]', str(th[2]))
            commands.append(text)

group_size = 5
cpu_idx = 0

for i in range(0, len(commands), group_size):
    group_cmds = commands[i:i + group_size]
    chain = " && ".join(group_cmds)

    cpu = allowed_cpus[cpu_idx % len(allowed_cpus)]
    print(f"taskset -c {cpu} bash -c '{chain}' &", end=' ')

    cpu_idx += 1


