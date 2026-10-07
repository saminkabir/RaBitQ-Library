
import svs as ps
import numpy as np
import math

dataset_path = '/home/cc/datasets/datasets/'

def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)



def get_data_gist(m=None):
    # data we will search through
    xb = read_vecs('gist/gist_base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('gist/gist_query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('gist/gist_groundtruth.ivecs')
    return xb,xq,gt

def get_data_sift(m=None):
    # data we will search through
    xb = read_vecs('siftsmall/siftsmall_base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('siftsmall/siftsmall_query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('siftsmall/siftsmall_groundtruth.ivecs')
    return xb,xq,gt

def get_data_sift1M(m=None):
    # data we will search through
    xb = read_vecs('sift/sift_base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('sift/sift_query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('sift/sift_groundtruth.ivecs')
    return xb,xq,gt


def get_data_generic(dataset):
    # data we will search through
    xb = read_vecs(dataset+'/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs(dataset+'/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs(dataset+'/groundtruth.ivecs')
    return xb,xq,gt
    

def get_data_glove(m=None):
    # data we will search through
    xb = read_vecs('glove/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('glove/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('glove/groundtruth.ivecs')
    return xb,xq,gt

def get_data_imageNet(m=None):
    # data we will search through
    xb = read_vecs('imageNet/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('imageNet/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('imageNet/groundtruth.ivecs')
    return xb,xq,gt

def get_data_notre(m=None):
    # data we will search through
    xb = read_vecs('notre/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('notre/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('notre/groundtruth.ivecs')
    return xb,xq,gt

def get_data_ukbench(m=None):
    # data we will search through
    xb = read_vecs('ukbench/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('ukbench/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('ukbench/groundtruth.ivecs')
    return xb,xq,gt

def get_data_crawl(m=None):
    # data we will search through
    xb = read_vecs('crawl/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('crawl/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('crawl/groundtruth.ivecs')
    return xb,xq,gt



def get_data_audio(m=None):
    # data we will search through
    xb = read_vecs('audio/audio_base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('audio/audio_query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('audio/audio_groundtruth.ivecs')
    return xb,xq,gt

def get_data_cifar(m=None):
    # data we will search through
    xb = read_vecs('cifar/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('cifar/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('cifar/groundtruth.ivecs')
    return xb,xq,gt

def extend_time_series_with_zeros(arr, dim):
    new_arr = np.zeros([arr.shape[0],dim])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j]=arr[i][j]
    return new_arr
    

def get_data_enron(m=None):
    # data we will search through
    xb = read_vecs('enron/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('enron/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('enron/groundtruth.ivecs')
    return xb,xq,gt






def get_data_millionSong(m=None):
    # data we will search through
    xb = read_vecs('millionSong/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('millionSong/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('millionSong/groundtruth.ivecs')
    return xb,xq,gt

def get_data_MNIST(m=None):
    # data we will search through
    xb = read_vecs('MNIST/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('MNIST/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('MNIST/groundtruth.ivecs')
    return xb,xq,gt

def get_data_nuswide(m=None):
    # data we will search through
    xb = read_vecs('nuswide/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('nuswide/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('nuswide/groundtruth.ivecs')
    return xb,xq,gt



def get_data_sun(m=None):
    # data we will search through
    xb = read_vecs('sun/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('sun/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('sun/groundtruth.ivecs')
    return xb,xq,gt

def get_data_deep1M(m=None):
    # data we will search through
    xb = read_vecs('deep/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('deep/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('deep/groundtruth.ivecs')
    return xb,xq,gt


def get_data_trevi(m=None):
    # data we will search through
    xb = read_vecs('trevi/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('trevi/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('trevi/groundtruth.ivecs')
    return xb,xq,gt


def get_data_uqv(m=None):
    # data we will search through
    xb = read_vecs('uqv/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('uqv/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('uqv/groundtruth.ivecs')
    return xb,xq,gt


def get_data_nytimes(m=None):
    # data we will search through
    xb = read_vecs('nytimes/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('nytimes/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('nytimes/groundtruth.ivecs')
    return xb,xq,gt

def get_data_lastfm(m=None):
    # data we will search through
    xb = read_vecs('lastfm/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('lastfm/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('lastfm/groundtruth.ivecs')
    return xb,xq,gt

def get_data_astro1m(m=None):
    # data we will search through
    xb = read_vecs('astro1m/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('astro1m/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('astro1m/groundtruth.ivecs')
    return xb,xq,gt

def get_data_seismic1m(m=None):
    # data we will search through
    xb = read_vecs('seismic1m/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('seismic1m/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('seismic1m/groundtruth.ivecs')
    return xb,xq,gt

def get_data_sald1m(m=None):
    # data we will search through
    xb = read_vecs('sald1m/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('sald1m/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('sald1m/groundtruth.ivecs')
    return xb,xq,gt


def get_data_bigann(m=None):
    # data we will search through
    xb = read_vecs('bigann/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('bigann/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('bigann/groundtruth.ivecs')
    return xb,xq,gt


def get_data_space1V(m=None):
    # data we will search through
    xb = read_vecs('space1V/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('space1V/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('space1V/groundtruth.ivecs')
    return xb,xq,gt




def get_data_text_to_image(m=None):
    # data we will search through
    xb = read_vecs('text-to-image/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('text-to-image/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('text-to-image/groundtruth.ivecs')
    return xb,xq,gt




def get_data_random(m=None):
    # data we will search through
    xb = read_vecs('random/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('random/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('random/groundtruth.ivecs')
    return xb,xq,gt

def get_data_movielens(m=None):
    # data we will search through
    xb = read_vecs('movielens/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('movielens/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('movielens/groundtruth.ivecs')
    return xb,xq,gt

def get_data_netflix(m=None):
    # data we will search through
    xb = read_vecs('netflix/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('netflix/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('netflix/groundtruth.ivecs')
    return xb,xq,gt



def get_data_tiny5m(m=None):
    # data we will search through
    xb = read_vecs('tiny5m/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('tiny5m/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('tiny5m/groundtruth.ivecs')
    return xb,xq,gt



def get_data_word2vec(m=None):
    # data we will search through
    xb = read_vecs('word2vec/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('word2vec/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('word2vec/groundtruth.ivecs')
    return xb,xq,gt
    return xb,xq,gt



def get_data_yahoomusic(m=None):
    # data we will search through
    xb = read_vecs('yahoomusic/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('yahoomusic/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('yahoomusic/groundtruth.ivecs')
    return xb,xq,gt



def get_data_znorm_gist(m=None):
    # data we will search through
    xb = read_vecs('gist/gist_base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('gist/gist_query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('gist/gist_groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_sift(m=None):
    # data we will search through
    xb = read_vecs('siftsmall/siftsmall_base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('siftsmall/siftsmall_query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('siftsmall/siftsmall_groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_sift1M(m=None):
    # data we will search through
    xb = read_vecs('sift/sift_base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('sift/sift_query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('sift/sift_groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_glove(m=None):
    # data we will search through
    xb = read_vecs('glove/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('glove/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('glove/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_imageNet(m=None):
    # data we will search through
    xb = read_vecs('imageNet/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('imageNet/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('imageNet/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_notre(m=None):
    # data we will search through
    xb = read_vecs('notre/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('notre/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('notre/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_ukbench(m=None):
    # data we will search through
    xb = read_vecs('ukbench/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('ukbench/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('ukbench/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_crawl(m=None):
    # data we will search through
    xb = read_vecs('crawl/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('crawl/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('crawl/groundtruth-z.ivecs')
    return xb,xq,gt



def get_data_znorm_audio(m=None):
    # data we will search through
    xb = read_vecs('audio/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('audio/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('audio/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_cifar(m=None):
    # data we will search through
    xb = read_vecs('cifar/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('cifar/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('cifar/groundtruth-z.ivecs')
    return xb,xq,gt

def extend_time_series_with_zeros(arr, dim):
    new_arr = np.zeros([arr.shape[0],dim])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j]=arr[i][j]
    return new_arr
    

def get_data_znorm_enron(m=None):
    # data we will search through
    xb = read_vecs('enron/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('enron/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('enron/groundtruth-z.ivecs')
    return xb,xq,gt






def get_data_znorm_millionSong(m=None):
    # data we will search through
    xb = read_vecs('millionSong/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('millionSong/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('millionSong/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_MNIST(m=None):
    # data we will search through
    xb = read_vecs('MNIST/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('MNIST/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('MNIST/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_nuswide(m=None):
    # data we will search through
    xb = read_vecs('nuswide/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('nuswide/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('nuswide/groundtruth-z.ivecs')
    return xb,xq,gt



def get_data_znorm_sun(m=None):
    # data we will search through
    xb = read_vecs('sun/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('sun/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('sun/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_deep1M(m=None):
    # data we will search through
    xb = read_vecs('deep/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('deep/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('deep/groundtruth-z.ivecs')
    return xb,xq,gt


def get_data_znorm_trevi(m=None):
    # data we will search through
    xb = read_vecs('trevi/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('trevi/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('trevi/groundtruth-z.ivecs')
    return xb,xq,gt


def get_data_znorm_uqv(m=None):
    # data we will search through
    xb = read_vecs('uqv/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('uqv/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('uqv/groundtruth-z.ivecs')
    return xb,xq,gt


def get_data_znorm_nytimes(m=None):
    # data we will search through
    xb = read_vecs('nytimes/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('nytimes/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('nytimes/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_lastfm(m=None):
    # data we will search through
    xb = read_vecs('lastfm/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('lastfm/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('lastfm/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_astro1m(m=None):
    # data we will search through
    xb = read_vecs('astro1m/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('astro1m/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('astro1m/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_seismic1m(m=None):
    # data we will search through
    xb = read_vecs('seismic1m/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('seismic1m/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('seismic1m/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_sald1m(m=None):
    # data we will search through
    xb = read_vecs('sald1m/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('sald1m/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('sald1m/groundtruth-z.ivecs')
    return xb,xq,gt


def get_data_znorm_bigann(m=None):
    # data we will search through
    xb = read_vecs('bigann/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('bigann/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('bigann/groundtruth-z.ivecs')
    return xb,xq,gt


def get_data_znorm_space1V(m=None):
    # data we will search through
    xb = read_vecs('space1V/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('space1V/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('space1V/groundtruth-z.ivecs')
    return xb,xq,gt




def get_data_znorm_text_to_image(m=None):
    # data we will search through
    xb = read_vecs('text-to-image/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('text-to-image/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('text-to-image/groundtruth-z.ivecs')
    return xb,xq,gt




def get_data_znorm_random(m=None):
    # data we will search through
    xb = read_vecs('random/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('random/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('random/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_movielens(m=None):
    # data we will search through
    xb = read_vecs('movielens/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('movielens/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('movielens/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_znorm_netflix(m=None):
    # data we will search through
    xb = read_vecs('netflix/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('netflix/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('netflix/groundtruth-z.ivecs')
    return xb,xq,gt



def get_data_znorm_tiny5m(m=None):
    # data we will search through
    xb = read_vecs('tiny5m/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('tiny5m/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('tiny5m/groundtruth-z.ivecs')
    return xb,xq,gt



def get_data_znorm_word2vec(m=None):
    # data we will search through
    xb = read_vecs('word2vec/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('word2vec/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('word2vec/groundtruth-z.ivecs')
    return xb,xq,gt
    return xb,xq,gt



def get_data_znorm_yahoomusic(m=None):
    # data we will search through
    xb = read_vecs('yahoomusic/base-z.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('yahoomusic/query-z.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('yahoomusic/groundtruth-z.ivecs')
    return xb,xq,gt

def get_data_ethz(m=None):
    # data we will search through
    xb = read_vecs('ethz/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('ethz/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('ethz/groundtruth.ivecs')
    return xb,xq,gt


def get_data_vcseis(m=None):
    # data we will search through
    xb = read_vecs('vcseis/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('vcseis/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('vcseis/groundtruth.ivecs')
    return xb,xq,gt


def get_data_txed(m=None):
    # data we will search through
    xb = read_vecs('txed/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('txed/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('txed/groundtruth.ivecs')
    return xb,xq,gt


def get_data_geofon(m=None):
    # data we will search through
    xb = read_vecs('geofon/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('geofon/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('geofon/groundtruth.ivecs')
    return xb,xq,gt


def get_data_lendb(m=None):
    # data we will search through
    xb = read_vecs('lendb/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('lendb/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('lendb/groundtruth.ivecs')
    return xb,xq,gt


def get_data_common(dataset):
    # data we will search through
    xb = read_vecs(dataset+'/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs(dataset+'/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs(dataset+'/groundtruth.ivecs').astype(np.int64)[:, :20]
    return xb,xq,gt


def get_data_stead(m=None):
    # data we will search through
    xb = read_vecs('stead/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('stead/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('stead/groundtruth.ivecs')
    return xb,xq,gt


def get_data_instancegm(m=None):
    # data we will search through
    xb = read_vecs('instancegm/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('instancegm/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('instancegm/groundtruth.ivecs')
    return xb,xq,gt


def get_data_Music(m=None):
    # data we will search through
    xb = read_vecs('Music/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('Music/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('Music/groundtruth.ivecs')
    return xb,xq,gt


def get_data_Yelp(m=None):
    # data we will search through
    xb = read_vecs('Yelp/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs('Yelp/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs('Yelp/groundtruth.ivecs')
    return xb,xq,gt


data_src_lvq_train = {
    'siftsm': '/data/kabir/similarity-search/dataset/siftsmall/siftsmall_base.fvecs',
    'sift1m': '/data/kabir/similarity-search/dataset/sift/sift_base.fvecs',
    'gist': '/data/kabir/similarity-search/dataset/gist/gist_base.fvecs',
    'notre': '/data/kabir/similarity-search/dataset/notre/base.fvecs',
    'ukbench': '/data/kabir/similarity-search/dataset/ukbench/base.fvecs',
    'audio': '/data/kabir/similarity-search/dataset/audio/audio_base.fvecs',
    'cifar': '/data/kabir/similarity-search/dataset/cifar/base.fvecs',
    'crawl': '/data/kabir/similarity-search/dataset/crawl/base.fvecs',
    'enron': '/data/kabir/similarity-search/dataset/enron/base.fvecs',
    'millionSong': '/data/kabir/similarity-search/dataset/millionSong/base.fvecs',
    'MNIST': '/data/kabir/similarity-search/dataset/MNIST/base.fvecs',
    'nuswide': '/data/kabir/similarity-search/dataset/nuswide/base.fvecs',
    'sun': '/data/kabir/similarity-search/dataset/sun/base.fvecs',
    'deep1m': '/data/kabir/similarity-search/dataset/deep/base.fvecs',
    'trevi': '/data/kabir/similarity-search/dataset/trevi/base.fvecs',
    'uqv': '/data/kabir/similarity-search/dataset/uqv/base.fvecs',
    'nytimes': '/data/kabir/similarity-search/dataset/nytimes/base.fvecs',
    'lastfm': '/data/kabir/similarity-search/dataset/lastfm/base.fvecs',
    'glove': '/data/kabir/similarity-search/dataset/glove/base.fvecs',
    'imageNet': '/data/kabir/similarity-search/dataset/imageNet/base.fvecs',
    'astro1m': '/data/kabir/similarity-search/dataset/astro1m/base.fvecs',
    'seismic1m': '/data/kabir/similarity-search/dataset/seismic1m/base.fvecs',
    'sald1m': '/data/kabir/similarity-search/dataset/sald1m/base.fvecs',
    'bigann': '/data/kabir/similarity-search/dataset/bigann/base.fvecs',
    'space1V': '/data/kabir/similarity-search/dataset/space1V/base.fvecs',
    'space': '/data/kabir/similarity-search/dataset/space1V/base.fvecs',
    'text-to-image': '/data/kabir/similarity-search/dataset/text-to-image/base.fvecs',
    'random': '/data/kabir/similarity-search/dataset/random/base.fvecs',
    'word2vec': '/data/kabir/similarity-search/dataset/word2vec/base.fvecs',
    'netflix': '/data/kabir/similarity-search/dataset/netflix/base.fvecs',
    'yahoomusic': '/data/kabir/similarity-search/dataset/yahoomusic/base.fvecs',
    'tiny5m': '/data/kabir/similarity-search/dataset/tiny5m/base.fvecs',
    'ethz': '/data/kabir/similarity-search/dataset/ethz/base.fvecs',
    'vcseis': '/data/kabir/similarity-search/dataset/vcseis/base.fvecs',
    'txed': '/data/kabir/similarity-search/dataset/txed/base.fvecs',
    'geofon': '/data/kabir/similarity-search/dataset/geofon/base.fvecs',
    'lendb': '/data/kabir/similarity-search/dataset/lendb/base.fvecs',
    'stead': '/data/kabir/similarity-search/dataset/stead/base.fvecs',
    'instancegm': '/data/kabir/similarity-search/dataset/instancegm/base.fvecs',
    'Music': '/data/kabir/similarity-search/dataset/Music/base.fvecs',
    'Yelp': '/data/kabir/similarity-search/dataset/Yelp/base.fvecs'
}

def get_compressed_loader(relative_path, _dim, data_loader = None):
    if data_loader != None:
        xb,xq,gt = data_loader()
        _dim = xb.shape[1]
    data_loader = ps.VectorDataLoader(
        relative_path,
        ps.DataType.float32,
        dims = _dim    # Passing dim is NOT optional in this case
    )
    padding = _dim
    compressed_loader = ps.LVQ8(
        data_loader,
        padding        # Passing padding is optional.
    )
    return compressed_loader

def getDatasetInfo(data_loader, dataset):
    xb, xq, gt = data_loader()
    x=xb
    train_summary=dict()
    train_summary['dataset']=dataset
    train_summary['type']='train'
    train_summary['sum']=np.sum(x,axis=0)
    train_summary['min']=np.min(x,axis=0)
    train_summary['max']=np.max(x,axis=0)
    train_summary['avg']=np.average(x,axis=0)
    train_summary['std']=np.std(x,axis=0)
    x=xq
    test_summary=dict()
    test_summary['dataset']=dataset
    test_summary['type']='test'
    test_summary['sum']=np.sum(x,axis=0)
    test_summary['min']=np.min(x,axis=0)
    test_summary['max']=np.max(x,axis=0)
    test_summary['avg']=np.average(x,axis=0)
    test_summary['std']=np.std(x,axis=0)
    return (train_summary,test_summary)

def getNumberOfDuplicateItems(data_loader, dataset):
    xb, xq, gt = data_loader()
    train_summary=dict()
    train_summary['dataset']=dataset
    duplicate_count=0
    for a in range (0,xb.shape[0]):
        for b in range (a+1,xb.shape[0]):
            x_a = xb[a]
            x_b = xb[b]
            minus = x_a - x_b
            distance = np.dot(minus.T, minus)
            if distance==0.0:
                duplicate_count=duplicate_count+1
    
    train_summary['duplicate_count']=duplicate_count
    return train_summary
                
    
    

# xb, xq, gt = get_data_cifar()
# print('cifer', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_enron()
# print('enron', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_millionSong()
# print('millionSong', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_MNIST()
# print('MNIST', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_nuswide()
# print('nuswide', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_sun()
# print('sun', xb.shape, xq.shape, gt.shape, xb.dtype)
# assert xb.dtype == np.float32

# xb, xq, gt = get_data_lastfm()
# print('lastfm', xb.shape, xq.shape, gt.shape)

# xb, xq, gt = get_data_random()
# print('random', xb.shape, xq.shape, gt.shape)

# xb, xq, gt = get_data_ethz()
# print('ethz', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_vcseis()
# print('vcseis', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_txed()
# print('txed', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_lendb()
# print('lendb', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_stead()
# print('stead', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_geofon()
# print('geofon', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_instancegm()
# print('instancegm', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_Music()
# print('Music', xb.shape, xq.shape, gt.shape)
# xb, xq, gt = get_data_Yelp()
# print('Yelp', xb.shape, xq.shape, gt.shape)


    
# astro1m (1000000, 256) (100, 256) (100, 100)
# audio (53387, 192) (200, 192) (200, 20)
# bigann (1000000, 128) (100, 128) (100, 100)
# cifar (50000, 512) (200, 512) (200, 20)
# crawl (1989995, 300) (10000, 300) (10000, 100)
# deep (1000000, 256) (200, 256) (200, 20)
# enron (94987, 1369) (200, 1369) (200, 20)
# gist (1000000, 960) (1000, 960) (1000, 100)
# glove (1192514, 100) (200, 100) (200, 20)
# imageNet (2340373, 150) (200, 150) (200, 20)
# lastfm (292385, 65) (100, 65) (100, 100)
# millionSong (992272, 420) (200, 420) (200, 20)
# movielens (10677, 150) (1000, 150) (1000, 100)
# MNIST (69000, 784) (200, 784) (200, 20)
# netflix (17770, 300) (1000, 300) (1000, 100)
# notre (332668, 128) (200, 128) (200, 20)
# nuswide (268643, 500) (200, 500) (200, 20)
# nytimes (290000, 256) (100, 256) (100, 100)
# random ((1000000, 100) (200, 100) (200, 20)
# sald1m (1000000, 128) (100, 128) (100, 100)
# seismic1m (1000000, 256) (100, 256) (100, 100)
# sift (1000000, 128) (10000, 128) (10000, 100)
# space1V (1000000, 100) (100, 100) (100, 100)
# sun (79106, 512) (200, 512) (200, 20)
# text-to-image (1000000, 200) (100, 200) (100, 100)
# tiny5m (5000000, 384) (1000, 384) (1000, 100)
# trevi (99900, 4096) (200, 4096) (200, 20)
# uqv (1000000, 256) (10000, 256) (10000, 100)
# ukbench (1097907, 128) (200, 128) (200, 20)
# word2vec (1000000, 300) (1000, 300) (1000, 100)
# yahoomusic (136736, 300) (100, 300) (100, 100)


# ethz (36643, 256) (100, 256) (100, 100)
# vcseis (160178, 256) (100, 256) (100, 100)
# txed (519589, 256) (100, 256) (100, 100)
# lendb (1000000, 256) (100, 256) (100, 100)
# stead (1000000, 256) (100, 256) (100, 100)
# geofon (275174, 128) (100, 128) (100, 100)
# instancegm (1000000, 128) (100, 128) (100, 100)


# Music (1000000, 100) (100, 100) (100, 100)
# Yelp (77079, 50) (100, 50) (100, 100)


[('ethz',36643,256,100),('vcseis',160178,256,100),('txed',519589,256,100),('lendb',1000000,256,100),('stead',1000000,256,100),('geofon',275174,128,100),('instancegm',1000000,128,100)]

[('Music',1000000,100,100),('Yelp',77079,50,100)]

# xb,xq,gt=get_data_generic('siftlarge')
# print('siftlarge', xb.shape, xq.shape, gt.shape)