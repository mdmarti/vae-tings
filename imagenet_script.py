from qp_vae_moons import QP_VAE
from b_vae_moons import B_VAE
from vae_moons import VAE
import matplotlib.pyplot as plt

import numpy as np
import os
import copy
import h5py
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_moons

import torch
import shutil
import gzip
import umap
import pickle
import getopt
import argparse
import itertools

import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torch.distributions import LowRankMultivariateNormal
from torch.distributions import multivariate_normal as mv

from qpvae_utils import get_qps_km, qpvae_ds, calc_dkl

def load_batch(datafolder, idx, categories,img_size=64):

    ##### categories is an np array

    dfile = os.path.join(datafolder, 'train_data_batch_')

    #d = unpickle(dfile + str(idx))
    with open(dfile + str(idx), 'rb') as pickle_file:
        d = pickle.load(pickle_file)

    x = d['data']
    y = d['labels']
    y = np.array([i - 1 for i in y])
    #print(x.shape)

    #x = x.reshape(img_size,img_size,64)
    x_out = []
    y_out = []
    for cat in categories:

        y_cat = y[y == cat]
        x_cat = x[y == cat, :]/np.float32(255)

        y_out.append(y_cat)
        x_out.append(x_cat)

    return np.vstack(x_out), np.hstack(y_out)



if __name__ == '__main__':

    root = '/home/mrmiews/Desktop/Pearson_Lab/imgdata'

    ds1_cats = np.arange(151,266,2)
    ds2_cats = np.arange(151,267,2)
    ds3_cats = np.arange(620,678,1)

    ds1_fname = os.path.join(root,'ds1.pickle')
    ds2_fname = os.path.join(root,'ds2.pickle')
    ds3_fname = os.path.join(root,'ds3.pickle')

    x1,y1 = [],[]
    x2,y2 = [],[]
    x3,y3 = [],[]
    print(os.path.isfile(ds1_fname))
    if not os.path.isfile(ds1_fname):
        for ii in range(1,11):
            #labels = load_batch(root, ii, ds1_cats)
            x1_tmp, y1_tmp = load_batch(root, ii, ds1_cats)
            x1.append(x1_tmp)
            y1.append(y1_tmp)

        x1, y1 = np.vstack(x1), np.hstack(y1)
        x1= np.reshape(x1, (-1,64,64,3))
        data = {'x':x1, 'y':y1}
        with open(ds1_fname,'wb') as f:
            pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(ds1_fname,'rb') as f:
            data = pickle.load(f)
        x1 = data['x']
        y1 = data['y']

    if not os.path.isfile(ds2_fname):
        for ii in range(1,11):
            #labels = load_batch(root, ii, ds1_cats)
            x2_tmp, y2_tmp = load_batch(root, ii, ds2_cats)
            x2.append(x2_tmp)
            y2.append(y2_tmp)

        x2, y2 = np.vstack(x2), np.hstack(y2)
        x2= np.reshape(x2, (-1,64,64,3))
        data = {'x':x2, 'y':y2}
        with open(ds2_fname,'wb') as f:
            pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(ds2_fname,'rb') as f:
            data = pickle.load(f)
        x2 = data['x']
        y2 = data['y']

    if not os.path.isfile(ds3_fname):
        for ii in range(1,11):
            #labels = load_batch(root, ii, ds1_cats)
            x3_tmp, y3_tmp = load_batch(root, ii, ds3_cats)
            x3.append(x3_tmp)
            y3.append(y3_tmp)

        x3, y3 = np.vstack(x3), np.hstack(y3)
        x3= np.reshape(x3, (-1,64,64,3))
        data = {'x':x3, 'y':y3}
        with open(ds3_fname,'wb') as f:
            pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(ds3_fname,'rb') as f:
            data = pickle.load(f)
        x3 = data['x']
        y3 = data['y']


    ##### Training set 1: ds1 only
    np.random.RandomState(13)
    n_dat1 = x1.shape[0]

    order1 = np.random.permutation(n_dat1)
    n_trx1 = np.round(n_dat1*0.6)
    n_tex1 = n_dat1 - n_trx1

    trord1 = order1[0:n_trx1]
    teord1 = order1[n_trx1::]
    tr_x1 = x1[trord1,:,:,:]
    te_x1 = x1[teord1,:,:,:]
    tr_y1 = y1[trord1]
    te_y1 = y1[teord1]
    ##### Training set 2: ds2 only
    n_dat2 = x2.shape[0]
    order2 = np.random.permutation(n_dat2)

    n_trx2 = np.round(n_dat2*0.6)
    n_tex2 = n_dat2 - n_trx2

    trord2 = order2[0:n_trx2]
    teord2 = order2[n_trx2::]
    tr_x2 = x2[trord2,:,:,:]
    te_x2 = x2[teord2,:,:,:]
    tr_y2 = y2[trord2]
    te_y2 = y2[teord2]
    ##### Training set 3: ds3 only
    n_dat3 = x3.shape[0]
    order3 = np.random.permutation(n_dat3)

    n_trx3 = np.round(n_dat3*0.6)
    n_tex3 = n_dat3 - n_trx3

    trord3 = order3[0:n_trx3]
    teord3 = order3[n_trx3::]
    tr_x3 = x3[trord3,:,:,:]
    te_x3 = x3[teord3,:,:,:]

    tr_y3 = y3[trord3]
    te_y3 = y3[teord3]
    ##### Training set 4: ds1, ds2

    x12 = np.concatenate((x1,x2),axis=0)
    y12 = np.hstack((y1,y2))

    n_dat12 = x12.shape[0]
    order12 = np.random.permutation(n_dat12)

    n_trx12 = np.round(n_dat12*0.6)
    n_tex12 = n_dat12 - n_trx12

    trord12 = order12[0:n_trx12]
    teord12 = order12[n_trx12::]
    tr_x12 = x12[trord12,:,:,:]
    te_x12 = x12[teord12,:,:,:]
    ##### Training set 5: ds1, ds3
    x13 = np.concatenate((x1,x3),axis=0)
    y13 = np.hstack((y1,y3))

    n_dat13 = x13.shape[0]
    order13 = np.random.permutation(n_dat13)

    n_trx13 = np.round(n_dat13*0.6)
    n_tex13 = n_dat13 - n_trx13

    trord13 = order13[0:n_trx13]
    teord12 = order13[n_trx13::]
    tr_x13 = x13[trord13,:,:,:]
    te_x13 = x13[teord13,:,:,:]


    ###### Setting up datastructures
    ds1_ds = qpvae_ds((x1,y1),transform = torch.FloatTensor)
    ds2_ds = qpvae_ds((x2,y2),transform = torch.FloatTensor)
    ds3_ds = qpvae_ds((x3,y3),transform = torch.FloatTensor)

    ds1dl = DataLoader(ds1_ds, batch_size = batch_size, \
                    shuffle = True,num_workers = num_workers)
    ds2dl = DataLoader(ds2_ds, batch_size = batch_size, \
                    shuffle = True,num_workers = num_workers)
        #print(labels)
