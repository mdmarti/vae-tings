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
from sklearn.linear_model import LinearRegression

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


def toy_model_script(n_train_runs = 10, root = '',figs=False):

    #root = '/home/mrmiews/Desktop/Pearson_Lab/models_moon'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    num_workers = min(7,os.cpu_count()-1)


    z_dim = 2

    mid = 20*np.sqrt(2)
    d1 = mv.MultivariateNormal(torch.tensor([-40.0,0.0,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))
    d2 = mv.MultivariateNormal(torch.tensor([-mid,mid,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))
    d3 = mv.MultivariateNormal(torch.tensor([0.0,40.0,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))
    d4 = mv.MultivariateNormal(torch.tensor([mid,mid,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))
    d5 = mv.MultivariateNormal(torch.tensor([40.0,0.0,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))
    d6 = mv.MultivariateNormal(torch.tensor([mid,-mid,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))
    d7 = mv.MultivariateNormal(torch.tensor([0.0,-40.0,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))
    d8 = mv.MultivariateNormal(torch.tensor([-mid,-mid,0.0,0.0,1],dtype=torch.float32),torch.eye(5,dtype=torch.float32))

    s1 = d1.sample([800,])
    s2 = d2.sample([800,])
    s3 = d3.sample([800,])
    s4 = d4.sample([800,])
    s5 = d5.sample([800,])
    s6 = d6.sample([800,])
    s7 = d7.sample([800,])
    s8 = d8.sample([800,])

    s1_lab = np.ones([800,])
    s2_lab = 2*np.ones([800,])
    s3_lab = 3*np.ones([800,])
    s4_lab = 4*np.ones([800,])
    s5_lab = 5*np.ones([800,])
    s6_lab = 6*np.ones([800,])
    s7_lab = 7*np.ones([800,])
    s8_lab = 8*np.ones([800,])

    batch_size = 100
    n_train1 = 800*4
    n_train2 = 800*4
    dsb =  torch.cat([s1,s2,s3,s4,s5,s6,s7,s8])
    ds1 = torch.cat([s1,s2,s3,s4])#,s5,s6,s7,s8])
    ds2 = torch.cat([s5,s6,s7,s8])


    labelsb = np.concatenate((s1_lab,s2_lab,s3_lab,s4_lab,\
                            s5_lab,s6_lab,s7_lab,s8_lab),axis=0)
    labels1 = np.concatenate((s1_lab,s2_lab,s3_lab,s4_lab),axis=0)
    labels2 =  np.concatenate((s5_lab,s6_lab,s7_lab,s8_lab),axis=0)

    ##### Plotting Actual Data #######

    ax = plt.gca()
    tmpDat = dsb.detach().cpu().numpy()
    for ii in range(4):
        td = tmpDat[labelsb == ii + 1,:]
        sel = np.random.choice(td.shape[0], 5)
        plt.scatter(td[sel,0],td[sel,1],marker='x',c='#666666',s=100.0)

    for ii in range(4,8):
        td = tmpDat[labelsb == ii + 1,:]
        sel = np.random.choice(td.shape[0], 5)
        plt.scatter(td[sel,0],td[sel,1],marker = 'd',c='#666666',s=100.0)
    plt.axis('off')
    plt.savefig(os.path.join(root,'data.png'))
    plt.close('all')

    ##### Joint Template gets this ###################
    train_dataset_all = qpvae_ds((list(dsb),list(labelsb)))


    ##### Sequential template gets ONE of these
    train_dataset1 = qpvae_ds((list(ds1), list(labels1)))
    train_dataset2 = qpvae_ds((list(ds2), list(labels2)))

    ##### Joint template gets this #########
    train_dataloader_all = DataLoader(train_dataset_all, batch_size = batch_size, \
                    shuffle = True,num_workers = num_workers)
    ######## Sequential template gets one of these
    train_dataloader_1 = DataLoader(train_dataset1, batch_size = batch_size, \
                    shuffle = True,num_workers = num_workers)
    train_dataloader_2 = DataLoader(train_dataset2, batch_size = batch_size, \
                    shuffle = True, num_workers = num_workers)
    ######## Dataloaders for evaluating latent space (MSE, logdet covar)
    latents_dataloader1 = DataLoader(train_dataset1, batch_size = batch_size, \
                    shuffle = False,num_workers = num_workers)
    latents_dataloader2 = DataLoader(train_dataset2, batch_size = batch_size, \
                    shuffle = False,num_workers = num_workers)

    ######## Might just need this one!!
    latents_dataloader_all = DataLoader(train_dataset_all, batch_size = batch_size, \
                    shuffle = False, num_workers = num_workers)

    ds1_dict = {'train':train_dataloader_1, 'test':train_dataloader_2}
    ds2_dict = {'train':train_dataloader_2, 'test':train_dataloader_1}
    dsb_dict = {'train': train_dataloader_all, 'test': train_dataloader_all}

    ##### Step 1: create template VAEs
    # first joint
    vae_template_joint_save = os.path.join(root,'vae_toy_joint_template')
    vae_template_joint = VAE(save_dir = vae_template_joint_save, z_dim = z_dim)
    joint_template_fname = os.path.join(vae_template_joint_save, 'checkpoint_200.tar')

    print('Training joint template VAE')
    if not os.path.isfile(joint_template_fname):
        vae_template_joint.train_loop(dsb_dict,epochs=201,test_freq=None,vis_freq=5)
    else:
        vae_template_joint.load_state(joint_template_fname)

    latents_1j = vae_template_joint.get_latent(latents_dataloader1)
    latents_2j = vae_template_joint.get_latent(latents_dataloader2)
    latents_aj = vae_template_joint.get_latent(latents_dataloader_all)

    # then sequential
    vae_template_seq_save = os.path.join(root,'vae_toy_seq_template')
    vae_template_seq = VAE(save_dir = vae_template_seq_save, z_dim = z_dim)
    seq_template_fname = os.path.join(vae_template_seq_save, 'checkpoint_200.tar')

    print('Training sequential template VAE')
    if not os.path.isfile(seq_template_fname):
        vae_template_seq.train_loop(ds1_dict,epochs=201,test_freq=None,vis_freq=5)
    else:
        vae_template_seq.load_state(seq_template_fname)

    latents_1s = vae_template_seq.get_latent(latents_dataloader1)
    latents_2s = vae_template_seq.get_latent(latents_dataloader2)
    latents_as = vae_template_seq.get_latent(latents_dataloader_all)


    n_qps = [2,4,8,16]#[64,128]
    n_qps.append(n_train1)


    ###### Get QPs from models ########
    ###### Only need these subsets, models will be trained on these qps + new ds #####
    old_datax1j, old_datay1j, old_dataz1j = get_qps_km(vae_template_joint, \
                                latents_1j, (list(ds1.detach().cpu().numpy()),list(labels1)),\
                                 n_qps, os.path.join(root,'latents1j'))

    old_datax1s, old_datay1s, old_dataz1s = get_qps_km(vae_template_seq, \
                                latents_1s, (list(ds1.detach().cpu().numpy()),list(labels1)),\
                                 n_qps, os.path.join(root,'latents1s'))

    ############# Create new datasets for training ##########
    qp_ds_copies = []

    for qp_ind, n_qp in enumerate(n_qps):
        ds2s_dict_copy = copy.copy(ds2_dict)
        ds2j_dict_copy = copy.copy(ds2_dict)

        oldx1j = old_datax1j[qp_ind]
        oldx1s = old_datax1s[qp_ind]

        oldy1j = old_datay1j[qp_ind]
        oldy1s = old_datay1s[qp_ind]

        oldz1j = old_dataz1j[qp_ind]
        oldz1s = old_dataz1s[qp_ind]

        train_dataset2j_copy = copy.copy(train_dataset2)
        train_dataset2s_copy = copy.copy(train_dataset2)

        train_dataset2j_copy.x = list(torch.cat((torch.stack(train_dataset2j_copy.x),oldx1j)))
        train_dataset2j_copy.y = np.hstack([train_dataset2j_copy.y,oldy1j])
        train_dataset2s_copy.x = list(torch.cat((torch.stack(train_dataset2s_copy.x),oldx1s)))
        train_dataset2s_copy.y = np.hstack([train_dataset2s_copy.y,oldy1s])

        tmp_dl2j = DataLoader(train_dataset2j_copy, batch_size = batch_size, \
            shuffle = True, num_workers=num_workers)
        tmp_dl2s = DataLoader(train_dataset2s_copy, batch_size = batch_size, \
            shuffle = True, num_workers=num_workers)

        ds2j_dict_copy['train'] = tmp_dl2j
        ds2s_dict_copy['train'] = tmp_dl2s

        qp_ds_copies.append((ds2j_dict_copy,ds2s_dict_copy))


    ############# Initialize output matrices ###############
    mse_matj = np.zeros((n_train1 + n_train2,n_train_runs,len(n_qps),3))
    mse_mats = np.zeros((n_train1 + n_train2,n_train_runs,len(n_qps),3))

    qpvj_latent_mat = np.zeros((n_train1 + n_train2, z_dim, len(n_qps),n_train_runs))
    qpvs_latent_mat = np.zeros((n_train1 + n_train2, z_dim, len(n_qps),n_train_runs))
    vj_latent_mat = np.zeros((n_train1 + n_train2, z_dim, len(n_qps),n_train_runs))
    vs_latent_mat = np.zeros((n_train1 + n_train2, z_dim, len(n_qps),n_train_runs))
    bvj_latent_mat = np.zeros((n_train1 + n_train2, z_dim, len(n_qps),n_train_runs))
    bvs_latent_mat = np.zeros((n_train1 + n_train2, z_dim, len(n_qps),n_train_runs))

    logdet_matj = np.zeros((n_train1 + n_train2,len(n_qps),3))
    logdet_mats = np.zeros((n_train1 + n_train2,len(n_qps),3))

    ############# Finally, train new models ################
    beta_bvae = 2.5

    for run_ind in range(n_train_runs):
        print('Beginning Training Run ', run_ind + 1)

        for qp_ind, n_qp in enumerate(n_qps):

            print('N QPs: ', n_qp)
            beta_qpvae = 15*batch_size/n_qp

            oldx1j = old_datax1j[qp_ind]
            oldx1s = old_datax1s[qp_ind]

            oldy1j = old_datay1j[qp_ind]
            oldy1s = old_datay1s[qp_ind]

            oldz1j = old_dataz1j[qp_ind]
            oldz1s = old_dataz1s[qp_ind]

            ds_dictj,ds_dicts = qp_ds_copies[qp_ind]

            vae_1j_anchor_save = os.path.join(root,'vae1j_toy_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            vae_1s_anchor_save = os.path.join(root,'vae1s_toy_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            bvae_1j_anchor_save = os.path.join(root,'bvae1j_toy_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            bvae_1s_anchor_save = os.path.join(root,'bvae1s_toy_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            qpvae_1j_anchor_save = os.path.join(root,'qpvae1j_toy_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            qpvae_1s_anchor_save = os.path.join(root,'qpvae1s_toy_run_'+str(run_ind) + '_nqp_' + str(n_qp))

            vae_1j_anchor_fn = os.path.join(vae_1j_anchor_save,'checkpoint_200.tar')
            vae_1s_anchor_fn = os.path.join(vae_1s_anchor_save,'checkpoint_200.tar')
            bvae_1j_anchor_fn = os.path.join(bvae_1j_anchor_save,'checkpoint_200.tar')
            bvae_1s_anchor_fn = os.path.join(bvae_1s_anchor_save,'checkpoint_200.tar')
            qpvae_1j_anchor_fn = os.path.join(qpvae_1j_anchor_save,'checkpoint_200.tar')
            qpvae_1s_anchor_fn = os.path.join(qpvae_1s_anchor_save,'checkpoint_200.tar')

            vae1j_anchor = VAE(save_dir = vae_1j_anchor_save, z_dim=z_dim)
            vae1s_anchor = VAE(save_dir = vae_1s_anchor_save, z_dim=z_dim)

            bvae1j_anchor = B_VAE(save_dir = bvae_1j_anchor_save, z_dim=z_dim, beta=beta_bvae)
            bvae1s_anchor = B_VAE(save_dir = bvae_1s_anchor_save, z_dim=z_dim, beta=beta_bvae)

            qpvae1j_anchor = QP_VAE(save_dir=qpvae_1j_anchor_save, z_dim=z_dim,no_sample = True,tau = beta_qpvae)
            qpvae1s_anchor = QP_VAE(save_dir=qpvae_1s_anchor_save, z_dim=z_dim,no_sample = True, tau = beta_qpvae)

            print('Training/Loading Vanilla VAEs')
            if not os.path.isfile(vae_1j_anchor_fn):
                vae1j_anchor.train_loop(ds_dictj,epochs=201,test_freq=None,vis_freq=10)
            else:
                vae1j_anchor.load_state(vae_1j_anchor_fn)
            if not os.path.isfile(vae_1s_anchor_fn):
                vae1s_anchor.train_loop(ds_dicts,epochs=201,test_freq=None,vis_freq=10)
            else:
                vae1s_anchor.load_state(vae_1s_anchor_fn)

            #### training/loading beta vaes
            print('Training/Loading Beta VAEs')
            if not os.path.isfile(bvae_1j_anchor_fn):
                bvae1j_anchor.train_loop(ds_dictj,epochs=201,test_freq=None,vis_freq=10)
            else:
                bvae1j_anchor.load_state(bvae_1j_anchor_fn)
            if not os.path.isfile(bvae_1s_anchor_fn):
                bvae1s_anchor.train_loop(ds_dicts,epochs=201,test_freq=None,vis_freq=10)
            else:
                bvae1s_anchor.load_state(bvae_1s_anchor_fn)


            #### training/loading qpvaes
            print('Training/Loading QP-VAEs')
            if not os.path.isfile(qpvae_1j_anchor_fn):
                qpvae1j_anchor.train_loop(ds2_dict,oldx1j, oldz1j,epochs=201,test_freq=None,vis_freq=10)
            else:
                qpvae1j_anchor.load_state(qpvae_1j_anchor_fn)
            if not os.path.isfile(qpvae_1s_anchor_fn):
                qpvae1s_anchor.train_loop(ds2_dict,oldx1s, oldz1s, epochs=201,test_freq=None,vis_freq=10)
            else:
                qpvae1s_anchor.load_state(qpvae_1s_anchor_fn)


            ############### Get Latents ###########
            vae1j_anchor_la = vae1j_anchor.get_latent(latents_dataloader_all)
            vae1s_anchor_la = vae1s_anchor.get_latent(latents_dataloader_all)

            bvae1j_anchor_la = bvae1j_anchor.get_latent(latents_dataloader_all)
            bvae1s_anchor_la = bvae1s_anchor.get_latent(latents_dataloader_all)

            qpvae1j_anchor_la = qpvae1j_anchor.get_latent(latents_dataloader_all)
            qpvae1s_anchor_la = qpvae1s_anchor.get_latent(latents_dataloader_all)

            ###### Save latents, so that we can find covariance matrix ########
            vj_latent_mat[:,:,qp_ind,run_ind] = vae1j_anchor_la
            vs_latent_mat[:,:,qp_ind,run_ind] = vae1s_anchor_la
            bvj_latent_mat[:,:,qp_ind,run_ind] = bvae1j_anchor_la
            bvs_latent_mat[:,:,qp_ind,run_ind] = bvae1s_anchor_la
            qpvj_latent_mat[:,:,qp_ind,run_ind] = qpvae1j_anchor_la
            qpvs_latent_mat[:,:,qp_ind,run_ind] = qpvae1s_anchor_la

            ####### Create linear models allowing for rotations, transfdata2ormations #####
            ###### Predicting original JOINT latent space from qp trained ########
            v1j_model = LinearRegression().fit(vae1j_anchor_la,latents_aj)
            v1s_model = LinearRegression().fit(vae1s_anchor_la,latents_aj)
            bv1j_model = LinearRegression().fit(bvae1j_anchor_la,latents_aj)
            bv1s_model = LinearRegression().fit(bvae1s_anchor_la,latents_aj)
            qpv1j_model = LinearRegression().fit(qpvae1j_anchor_la,latents_aj)
            qpv1s_model = LinearRegression().fit(qpvae1s_anchor_la,latents_aj)

            pred_v1j = v1j_model.predict(vae1j_anchor_la)
            pred_v1s = v1s_model.predict(vae1s_anchor_la)
            pred_bv1j = bv1j_model.predict(bvae1j_anchor_la)
            pred_bv1s = bv1s_model.predict(bvae1s_anchor_la)
            pred_qpv1j = qpv1j_model.predict(qpvae1j_anchor_la)
            pred_qpv1s = qpv1s_model.predict(qpvae1s_anchor_la)

            mse_matj[:,run_ind,qp_ind,0] = np.sum((pred_v1j - latents_aj)**2,axis=1)
            mse_mats[:,run_ind,qp_ind,0] = np.sum((pred_v1s - latents_aj)**2,axis=1)

            mse_matj[:,run_ind,qp_ind,1] = np.sum((pred_bv1j - latents_aj)**2,axis=1)
            mse_mats[:,run_ind,qp_ind,1] = np.sum((pred_bv1s - latents_aj)**2,axis=1)

            mse_matj[:,run_ind,qp_ind,2] = np.sum((pred_qpv1j - latents_aj)**2,axis=1)
            mse_mats[:,run_ind,qp_ind,2] = np.sum((pred_qpv1s - latents_aj)**2,axis=1)


            if figs:
                '''
                To-do: implement figure-making
                '''
                pass

    for lat_ind in range(n_train1 + n_train2):
        for qp_ind, n_qp in enumerate(n_qps):
            covar_latent_vj = np.cov(vj_latent_mat[lat_ind,:,qp_ind,:])
            _,ldvj = np.linalg.slogdet(covar_latent_vj)

            covar_latent_vs = np.cov(vs_latent_mat[lat_ind,:,qp_ind,:])
            _,ldvs = np.linalg.slogdet(covar_latent_vs)

            covar_latent_bvj = np.cov(bvj_latent_mat[lat_ind,:,qp_ind,:])
            _,ldbvj = np.linalg.slogdet(covar_latent_bvj)

            covar_latent_bvs = np.cov(bvs_latent_mat[lat_ind,:,qp_ind,:])
            _,ldbvs = np.linalg.slogdet(covar_latent_bvs)

            covar_latent_qpvj = np.cov(qpvj_latent_mat[lat_ind,:,qp_ind,:])
            _,ldqpvj = np.linalg.slogdet(covar_latent_qpvj)

            covar_latent_qpvs = np.cov(qpvs_latent_mat[lat_ind,:,qp_ind,:])
            _,ldqpvs = np.linalg.slogdet(covar_latent_qpvs)



            logdet_matj[lat_ind,qp_ind,0] = ldvj
            logdet_matj[lat_ind,qp_ind,1] = ldbvj
            logdet_matj[lat_ind,qp_ind,2] = ldqpvj

            logdet_mats[lat_ind,qp_ind,0] = ldvs
            logdet_mats[lat_ind,qp_ind,1] = ldbvs
            logdet_mats[lat_ind,qp_ind,2] = ldqpvs

    return mse_matj,logdet_matj,mse_mats,logdet_mats

if __name__ == '__main__':

    pass
