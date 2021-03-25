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


if __name__ == '__main__':

    root = '/home/mrmiews/Desktop/Pearson_Lab/models_moon'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    num_workers = min(7,os.cpu_count()-1)

    n_qps = [16,32,64,128]#[64,128]

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

    t1 = d1.sample([400,])
    t2 = d2.sample([400,])
    t3 = d3.sample([400,])
    t4 = d4.sample([400,])
    t5 = d5.sample([400,])
    t6 = d6.sample([400,])
    t7 = d7.sample([400,])
    t8 = d8.sample([400,])

    t1_lab = np.ones([400,])
    t2_lab = 2*np.ones([400,])
    t3_lab = 3*np.ones([400,])
    t4_lab = 4*np.ones([400,])
    t5_lab = 5*np.ones([400,])
    t6_lab = 6*np.ones([400,])
    t7_lab = 7*np.ones([400,])
    t8_lab = 8*np.ones([400,])

    batch_size = 100
    n_train1 = 800*4
    n_train0 = 800*4
    data12 =  torch.cat([s1,s2,s3,s4,s5,s6,s7,s8])
    data1 = torch.cat([s1,s2,s3,s4])#,s5,s6,s7,s8])
    data2 = torch.cat([s5,s6,s7,s8])




    labels12 = np.concatenate((s1_lab,s2_lab,s3_lab,s4_lab,\
                            s5_lab,s6_lab,s7_lab,s8_lab),axis=0)
    labels1 = np.concatenate((s1_lab,s2_lab,s3_lab,s4_lab),axis=0)
    labels2 =  np.concatenate((s5_lab,s6_lab,s7_lab,s8_lab),axis=0)

    ax = plt.gca()
    tmpDat = data12.detach().cpu().numpy()
    for ii in range(8):
        plt.plot(tmpDat[labels12 == ii + 1,0],tmpDat[labels12 == ii + 1,1])

    plt.savefig('data.png')
    plt.close('all')


    train_dataset_all = qpvae_ds((list(data12),list(labels12)))
    train_dataset0 = qpvae_ds((list(data1), list(labels1)))
    train_dataset1 = qpvae_ds((list(data2), list(labels2)))
    '''
    print(np.unique(np.array(train_dataset0.y)))
    print(np.unique(np.array(train_dataset1.y)))
    '''
    train_all_dataloader = DataLoader(train_dataset_all, batch_size = batch_size, \
                    shuffle = True,num_workers = num_workers)
    train_across_dataloader0 = DataLoader(train_dataset0, batch_size = batch_size, \
                    shuffle = True,num_workers = num_workers)
    latents_dataloader0 = DataLoader(train_dataset0, batch_size = batch_size, \
                    shuffle = False,num_workers = num_workers)
    train_across_dataloader1 = DataLoader(train_dataset1, batch_size = batch_size, \
                    shuffle = True,num_workers = num_workers)
    latents_dataloader1 = DataLoader(train_dataset1, batch_size = batch_size, \
                    shuffle = False,num_workers = num_workers)

    latents_dataloader_all = DataLoader(train_dataset_all, batch_size = batch_size, \
                    shuffle = False, num_workers = num_workers)

    across1 = {'train':train_across_dataloader1, 'test':train_across_dataloader0}
    across0 = {'train':train_across_dataloader0, 'test':train_across_dataloader1}
    across_all = {'train': train_all_dataloader, 'test': train_all_dataloader}


    #### Step 1: Create template VAEs
    #Train one on ds1, one on ds2, one on both, then obtain latents from all

    vae_template_all_save = os.path.join(root, 'vae_moons_both_template')
    vae_template0_save = os.path.join(root,'vae_moons_template_0')
    vae_template1_save = os.path.join(root,'vae_moons_template_1')


    vae_templateb = VAE(save_dir=vae_template_all_save,z_dim = z_dim)
    vae_template0 = VAE(save_dir = vae_template0_save, z_dim = z_dim)
    vae_template1 = VAE(save_dir = vae_template1_save, z_dim = z_dim)

    template0_fname = os.path.join(vae_template0_save, 'checkpoint_200.tar')
    template1_fname = os.path.join(vae_template1_save, 'checkpoint_200.tar')
    templateb_fname = os.path.join(vae_template_all_save, 'checkpoint_200.tar')

    print('Training template VAE: DS 1')
    if not os.path.isfile(template0_fname):
        vae_template0.train_loop(across0,epochs=201,test_freq=None,vis_freq=5)
    else:
        vae_template0.load_state(template0_fname)

    print('Training template VAE: DS 2')
    if not os.path.isfile(template1_fname):
        vae_template1.train_loop(across1,epochs=201,test_freq=None,vis_freq=5)
    else:
        vae_template1.load_state(template1_fname)

    print('Training template VAE: DS 1,2')
    if not os.path.isfile(templateb_fname):
        vae_templateb.train_loop(across_all,epochs=201,test_freq=None,vis_freq=5)
    else:
        vae_templateb.load_state(templateb_fname)


    latents11 = vae_template1.get_latent(latents_dataloader1)
    latents00 = vae_template0.get_latent(latents_dataloader0)

    latents1b = vae_templateb.get_latent(latents_dataloader1)
    latents0b = vae_templateb.get_latent(latents_dataloader0)

    latents_a1 = vae_template1.get_latent(latents_dataloader_all)
    latents_a0 = vae_template0.get_latent(latents_dataloader_all)
    latents_ab = vae_templateb.get_latent(latents_dataloader_all)


    ###### Get latents from each model ######
    old_datax1_1, old_datay1_1, old_dataz1_1, qps1_1 = get_qps_km(vae_template1, \
                                latents11, (list(data2.detach().cpu().numpy()),list(labels2)),\
                                 n_qps, os.path.join(root,'latents11'))
    old_datax0_0, old_datay0_0, old_dataz0_0, qps0_0 = get_qps_km(vae_template0, \
                                latents00, (list(data1.detach().cpu().numpy()),list(labels1)), \
                                n_qps, os.path.join(root,'latents00'))

    old_datax1_b, old_datay1_b, old_dataz1_b, qps1_b = get_qps_km(vae_templateb, \
                                latents1b, (list(data2.detach().cpu().numpy()),list(labels2)),\
                                 n_qps, os.path.join(root,'latents1b'))
    old_datax0_b, old_datay0_b, old_dataz0_b, qps0_b = get_qps_km(vae_templateb, \
                                latents0b, (list(data1.detach().cpu().numpy()),list(labels1)), \
                                n_qps, os.path.join(root,'latents0b'))

    ax = plt.gca()
    #print(old_dataz1_1[0].shape)
    #print(qps0_b.shape)
    scat1 = ax.scatter(latents0b[:,0], latents0b[:,1],c='dodgerblue',alpha=0.05)
    #print(qps0_b)
    #print(qps1_b)
    #print(qps1_1)
    #print(qps0_0)
    for z in old_dataz0_b:
        #print(z)
        z = z.detach().cpu().numpy()
        #print(qp.shape)
        scat2 = ax.scatter(z[:,0],z[:,1], c='r')

    plt.legend([scat1,scat2],['latents','qps'])

    plt.savefig('stuff.png')
    across11_copy = copy.copy(across1)
    across00_copy = copy.copy(across0)
    across1b_copy = copy.copy(across1)
    across0b_copy = copy.copy(across0)

    latent_inds1 = np.repeat(np.array([1,2,3]),n_train1)#,4
    latent_inds1 = np.hstack([np.zeros([n_train1,]),latent_inds1])
    latent_inds0 = np.repeat(np.array([1,2,3]),n_train0)#,4
    latent_inds0 = np.hstack([np.zeros([n_train0,]),latent_inds0])

    latent_indsa = np.repeat(np.array([1,2,3]),n_train0 + n_train1)
    latent_indsa = np.hstack([np.zeros([n_train0 + n_train1,]),latent_indsa])

    models = [1,2,3]#,4]
    mnames = ['Vanilla', 'bVAE','qpVAE']#'vqVAE',
    color_orig = 'dodgerblue'
    model_colors = ['lime','brown','darkblue','crimson']

    beta_bvae = 2.5

    for qp_ind, n_qp in enumerate(n_qps):

        beta_qpvae = 20*batch_size/n_qp

        print('Training on N QPs = ', n_qp)
        oldx1_1 = old_datax1_1[qp_ind]
        oldx0_0 = old_datax0_0[qp_ind]
        oldx1_b = old_datax1_b[qp_ind]
        oldx0_b = old_datax0_b[qp_ind]
        #print(oldx1.shape)

        oldy1_1 = old_datay1_b[qp_ind]
        oldy0_0 = old_datay0_b[qp_ind]
        oldy1_b = old_datay1_b[qp_ind]
        oldy0_b = old_datay0_b[qp_ind]

        oldz1_1 = old_dataz1_1[qp_ind]
        oldz0_0 = old_dataz0_0[qp_ind]
        oldz1_b = old_dataz1_b[qp_ind]
        oldz0_b = old_dataz0_b[qp_ind]

        train_dataset11_copy = copy.copy(train_dataset1)
        train_dataset00_copy = copy.copy(train_dataset0)
        train_dataset1b_copy = copy.copy(train_dataset1)
        train_dataset0b_copy = copy.copy(train_dataset0)

        train_dataset11_copy.x = list(torch.cat((torch.stack(train_dataset11_copy.x),torch.FloatTensor(oldx0_0))))
        train_dataset11_copy.y = np.hstack([train_dataset11_copy.y,oldy0_0])
        train_dataset00_copy.x = list(torch.cat((torch.stack(train_dataset00_copy.x),torch.FloatTensor(oldx1_1))))
        train_dataset00_copy.y = np.hstack([train_dataset00_copy.y,oldy1_1])

        train_dataset1b_copy.x = list(torch.cat((torch.stack(train_dataset1b_copy.x),torch.FloatTensor(oldx0_b))))
        train_dataset1b_copy.y = np.hstack([train_dataset1b_copy.y,oldy0_b])
        train_dataset0b_copy.x = list(torch.cat((torch.stack(train_dataset0b_copy.x),torch.FloatTensor(oldx1_b))))
        train_dataset0b_copy.y = np.hstack([train_dataset0b_copy.y,oldy1_b])
        '''
        ax = plt.gca()
        tmpx0 = torch.stack(train_dataset0_copy.x).detach().cpu().numpy()
        tmpx1 = torch.stack(train_dataset1_copy.x).detach().cpu().numpy()

        for ii in range(8):
            if ii < 4:
                labels = train_dataset0_copy.y == ii+1
                tmpDat = tmpx0[labels,:]
            else:
                label = train_dataset1_copy.y == ii+1
                tmpDat = tmpx1[labels,:]

            plt.plot(tmpDat[:,0],tmpDat[:,1])

        plt.savefig(os.path.join(root,'data2.png'))
        plt.close('all')
        '''
        tmp_dl11 = DataLoader(train_dataset11_copy, batch_size = batch_size, \
            shuffle = True, num_workers=num_workers)
        tmp_dl00 = DataLoader(train_dataset00_copy, batch_size = batch_size, \
            shuffle = True, num_workers=num_workers)

        tmp_dl1b = DataLoader(train_dataset1b_copy, batch_size = batch_size, \
            shuffle = True, num_workers=num_workers)
        tmp_dl0b = DataLoader(train_dataset0b_copy, batch_size = batch_size, \
            shuffle = True, num_workers=num_workers)

        across11_copy['train'] = tmp_dl11
        across00_copy['train'] = tmp_dl00

        across1b_copy['train'] = tmp_dl1b
        across0b_copy['train'] = tmp_dl0b

        vae_save11 = os.path.join(root,'vae_moon_' + str(n_qp) + 'qp11')
        vae_save00 = os.path.join(root,'vae_moon_' + str(n_qp) + 'qp00')
        vae_save1b = os.path.join(root,'vae_moon_' + str(n_qp) + 'qp1b')
        vae_save0b = os.path.join(root,'vae_moon_' + str(n_qp) + 'qp0b')

        bvae_save11 = os.path.join(root,'bvae_moon_' + str(n_qp) + 'qp11')
        bvae_save00 = os.path.join(root,'bvae_moon_' + str(n_qp) + 'qp00')
        bvae_save1b = os.path.join(root,'bvae_moon_' + str(n_qp) + 'qp1b')
        bvae_save0b = os.path.join(root,'bvae_moon_' + str(n_qp) + 'qp0b')

        qpvae_save11 = os.path.join(root,'qpvae_moon_' + str(n_qp) + 'qp11')
        qpvae_save00 = os.path.join(root,'qpvae_moon_' + str(n_qp) + 'qp00')
        qpvae_save1b = os.path.join(root,'qpvae_moon_' + str(n_qp) + 'qp1b')
        qpvae_save0b = os.path.join(root,'qpvae_moon_' + str(n_qp) + 'qp0b')

        vaes11 = VAE(save_dir = vae_save11, z_dim=z_dim)
        vaes00 = VAE(save_dir = vae_save00, z_dim=z_dim)
        vaes1b = VAE(save_dir = vae_save1b, z_dim=z_dim)
        vaes0b = VAE(save_dir = vae_save0b, z_dim=z_dim)
        #dummyvae = VAE(save_dir = vae_save1, z_dim=z_dim)

        bvaes11 = B_VAE(save_dir = bvae_save11, z_dim=z_dim, beta=beta_bvae)
        bvaes00 = B_VAE(save_dir = bvae_save00, z_dim=z_dim, beta=beta_bvae)
        bvaes1b = B_VAE(save_dir = bvae_save1b, z_dim=z_dim, beta=beta_bvae)
        bvaes0b = B_VAE(save_dir = bvae_save0b, z_dim=z_dim, beta=beta_bvae)
        #dummybvae = B_VAE(save_dir = bvae_save2, z_dim=z_dim, beta=beta_bvae)

        qpvaes11 = QP_VAE(save_dir=qpvae_save11, z_dim=z_dim,no_sample = True,tau = beta_qpvae)
        qpvaes00 = QP_VAE(save_dir=qpvae_save00, z_dim=z_dim,no_sample = True, tau = beta_qpvae)
        qpvaes1b = QP_VAE(save_dir=qpvae_save1b, z_dim=z_dim,no_sample = True,tau = beta_qpvae)
        qpvaes0b = QP_VAE(save_dir=qpvae_save0b, z_dim=z_dim,no_sample = True, tau = beta_qpvae)

        vae11_filename = os.path.join(vae_save11, 'checkpoint_200.tar')
        vae00_filename = os.path.join(vae_save00, 'checkpoint_200.tar')
        vae1b_filename = os.path.join(vae_save1b, 'checkpoint_200.tar')
        vae0b_filename = os.path.join(vae_save0b, 'checkpoint_200.tar')

        bvae11_filename = os.path.join(bvae_save11, 'checkpoint_200.tar')
        bvae00_filename = os.path.join(bvae_save00, 'checkpoint_200.tar')
        bvae1b_filename = os.path.join(bvae_save1b, 'checkpoint_200.tar')
        bvae0b_filename = os.path.join(bvae_save0b, 'checkpoint_200.tar')

        qpvae11_filename = os.path.join(qpvae_save11, 'checkpoint_200.tar')
        qpvae00_filename = os.path.join(qpvae_save00, 'checkpoint_200.tar')
        qpvae1b_filename = os.path.join(qpvae_save1b, 'checkpoint_200.tar')
        qpvae0b_filename = os.path.join(qpvae_save0b, 'checkpoint_200.tar')

        #### training/loading vanilla vaes ####
        print('Training/Loading Vanilla VAEs')
        if not os.path.isfile(vae11_filename):
            vaes11.train_loop(across11_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            vaes11.load_state(vae11_filename)
        if not os.path.isfile(vae00_filename):
            vaes00.train_loop(across00_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            vaes00.load_state(vae00_filename)
        if not os.path.isfile(vae1b_filename):
            vaes1b.train_loop(across1b_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            vaes1b.load_state(vae1b_filename)
        if not os.path.isfile(vae0b_filename):
            vaes0b.train_loop(across0b_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            vaes0b.load_state(vae0b_filename)

        #### training/loading beta vaes
        print('Training/Loading Beta VAEs')
        if not os.path.isfile(bvae11_filename):
            bvaes11.train_loop(across11_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            bvaes11.load_state(bvae11_filename)
        if not os.path.isfile(bvae00_filename):
            bvaes00.train_loop(across00_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            bvaes00.load_state(bvae00_filename)
        if not os.path.isfile(bvae1b_filename):
            bvaes1b.train_loop(across1b_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            bvaes1b.load_state(bvae1b_filename)
        if not os.path.isfile(bvae0b_filename):
            bvaes0b.train_loop(across0b_copy,epochs=201,test_freq=None,vis_freq=10)
        else:
            bvaes0b.load_state(bvae0b_filename)

        #print('Training/Loading VQVAEs')
        '''
        if not os.path.isfile(vqvae1_filename):
            vqvaes1[qp_ind].train_loop(across1_copy,epochs=201,test_freq=None,vis_freq=5)
        else:
            vqvaes1[qp_ind].load_state(vqvae1_filename)
            vqvaes1[qp_ind].train_loop(across1_copy,epochs=80,test_freq=None,vis_freq=5)
        if not os.path.isfile(vqvae2_filename):
            vqvaes2[qp_ind].train_loop(across2_copy,epochs=201,test_freq=None,vis_freq=5)
        else:
            vqvaes2[qp_ind].load_state(vqvae2_filename)
        '''


        #### training/loading qpvaes
        print('Training/Loading QP-VAEs')
        if not os.path.isfile(qpvae11_filename):
            qpvaes11.train_loop(across1,oldx0_0, oldz0_0,epochs=201,test_freq=None,vis_freq=10)
        else:
            qpvaes11.load_state(qpvae11_filename)
        if not os.path.isfile(qpvae00_filename):
            qpvaes00.train_loop(across0,oldx1_1, oldz1_1, epochs=201,test_freq=None,vis_freq=10)
        else:
            qpvaes00.load_state(qpvae00_filename)
        if not os.path.isfile(qpvae1b_filename):
            qpvaes1b.train_loop(across1,oldx0_b, oldz0_b,epochs=201,test_freq=None,vis_freq=10)
        else:
            qpvaes11.load_state(qpvae1b_filename)
        if not os.path.isfile(qpvae0b_filename):
            qpvaes0b.train_loop(across0,oldx1_b, oldz1_b, epochs=201,test_freq=None,vis_freq=10)
        else:
            qpvaes0b.load_state(qpvae0b_filename)


    #    dummyqpvae.load_state(qpvae1_filename)
        ######### Once models have been trkl_l1_1_2ained, we can compare their latent spaces
        ######### What are the right comparisons here?

        ######### what if we train a beta vae then compress it
        ######### can this learn the same disentangled latent space?


        ########regardless, compare latents1 to model trained on latents2, given latents1
        #use latents_dataloader1, latents_dataloader2

        #latents = model.get_latent(latents_dataloader)


        print('Getting vanilla latents')
        vanilla_m11l0 = vaes11.get_latent(latents_dataloader0)
        vanilla_m11l1 = vaes11.get_latent(latents_dataloader1)

        vanilla_m00l1 = vaes00.get_latent(latents_dataloader1)
        vanilla_m00l0 = vaes00.get_latent(latents_dataloader0)

        vanilla_m1bl1 = vaes1b.get_latent(latents_dataloader1)
        vanilla_m1bl0 = vaes1b.get_latent(latents_dataloader0)

        vanilla_m0bl1 = vaes0b.get_latent(latents_dataloader1)
        vanilla_m0bl0 = vaes0b.get_latent(latents_dataloader0)

        vanilla_m11la = vaes11.get_latent(latents_dataloader_all)
        vanilla_m1bla = vaes1b.get_latent(latents_dataloader_all)
        vanilla_m00la = vaes00.get_latent(latents_dataloader_all)
        vanilla_m0bla = vaes0b.get_latent(latents_dataloader_all)

        print('Getting Beta latents')
        bvae_m11l0 = bvaes11.get_latent(latents_dataloader0)
        bvae_m11l1 = bvaes11.get_latent(latents_dataloader1)

        bvae_m00l1 = bvaes00.get_latent(latents_dataloader1)
        bvae_m00l0 = bvaes00.get_latent(latents_dataloader0)

        bvae_m1bl0 = bvaes1b.get_latent(latents_dataloader0)
        bvae_m1bl1 = bvaes1b.get_latent(latents_dataloader1)

        bvae_m0bl1 = bvaes0b.get_latent(latents_dataloader1)
        bvae_m0bl0 = bvaes0b.get_latent(latents_dataloader0)

        bvae_m11la = bvaes11.get_latent(latents_dataloader_all)
        bvae_m1bla = bvaes1b.get_latent(latents_dataloader_all)
        bvae_m00la = bvaes00.get_latent(latents_dataloader_all)
        bvae_m0bla = bvaes0b.get_latent(latents_dataloader_all)

        '''
        vqvae_m1l2 = vqvaes1[qp_ind].get_latent(latents_dataloader2)
        vqvae_m2l1 = vqvaes2[qp_ind].get_latent(latents_dataloader1)
        '''
        print('Getting qpVAE latents')
        qpvaes_m11l0 = qpvaes11.get_latent(latents_dataloader0)
        qpvaes_m11l1 = qpvaes11.get_latent(latents_dataloader1)

        qpvaes_m00l1 = qpvaes00.get_latent(latents_dataloader1)
        qpvaes_m00l0 = qpvaes00.get_latent(latents_dataloader0)

        qpvaes_m1bl0 = qpvaes1b.get_latent(latents_dataloader0)
        qpvaes_m1bl1 = qpvaes1b.get_latent(latents_dataloader1)

        qpvaes_m0bl1 = qpvaes0b.get_latent(latents_dataloader1)
        qpvaes_m0bl0 = qpvaes0b.get_latent(latents_dataloader0)

        qpvaes_m11la = qpvaes11.get_latent(latents_dataloader_all)
        qpvaes_m1bla = qpvaes1b.get_latent(latents_dataloader_all)
        qpvaes_m00la = qpvaes00.get_latent(latents_dataloader_all)
        qpvaes_m0bla = qpvaes0b.get_latent(latents_dataloader_all)


        print('Done!')
        all_l_a11 = np.vstack([latents_a1, vanilla_m11la, bvae_m11la, qpvaes_m11la])
        all_l_a00 = np.vstack([latents_a0, vanilla_m00la, bvae_m00la, qpvaes_m00la])
        all_l_a1b = np.vstack([latents_ab, vanilla_m1bla, bvae_m1bla, qpvaes_m1bla])
        all_l_a0b = np.vstack([latents_ab, vanilla_m0bla, bvae_m0bla, qpvaes_m0bla])

        ## 11-11 comparisons: trained with 11 qps, compared to 11 latents
        all_l_1111 = np.vstack([latents11,vanilla_m00l1,bvae_m00l1,qpvaes_m00l1])
        ## 11-00 comparisons: trained with 00 qps, compared to 11 latents
        all_l_1100 = np.vstack([latents11,vanilla_m11l1,bvae_m11l1,qpvaes_m11l1])
        ## 1b-1b comparisons: trained with 1b qps, compared to 1b latents
        all_l_1b1b = np.vstack([latents1b,vanilla_m0bl1,bvae_m0bl1,qpvaes_m0bl1])
        ## 1b-0b comparisons: trained with 0b qps, compared to 1b latents
        all_l_1b0b = np.vstack([latents1b,vanilla_m0bl1,bvae_m0bl1,qpvaes_m0bl1])
        ## 1b-11 comparisons: trained with 11 qps, compared to 1b latents
        all_l_1b11 = np.vstack([latents1b,vanilla_m00l1,bvae_m00l1,qpvaes_m00l1])
        ## 1b-00 comparisons: trained with 00 qps, compared to 1b latents
        all_l_1b00 = np.vstack([latents1b,vanilla_m11l1,bvae_m11l1,qpvaes_m11l1])

        ## 00-00 comparisons: trained with 00 qps, compared to 00 latents
        all_l_0000 = np.vstack([latents00,vanilla_m11l0,bvae_m11l0,qpvaes_m11l0])
        ## 00-11 comparisons: trained with 11 qps, compared to 00 latents
        all_l_0011 = np.vstack([latents00,vanilla_m00l0,bvae_m00l0,qpvaes_m00l0])
        ## 0b-0b comparisons: trained with 0b qps, compared to 0b latents
        all_l_0b0b = np.vstack([latents0b,vanilla_m1bl0,bvae_m1bl0,qpvaes_m1bl0])
        ## 0b-1b comparisons: trained with 1b qps, compared to 0b latents
        all_l_0b1b = np.vstack([latents0b,vanilla_m0bl0,bvae_m0bl0,qpvaes_m0bl0])
        ## 0b-00 comparisons: trained with 00 qps, compared to 0b latents
        all_l_0b00 = np.vstack([latents0b,vanilla_m11l0,bvae_m11l0,qpvaes_m11l0])
        ## 0b-11 comparisons: trained with 11 qps, compared to 0b latents
        all_l_0b11 = np.vstack([latents0b,vanilla_m00l0,bvae_m00l0,qpvaes_m00l0])

        '''
        print('Making plots for all models')


        for model in models:

            mname = mnames[model-1]


            orig_inds1 = latent_inds1 == 0
            orig_inds0 = latent_inds0 == 0
            inds1 = latent_inds1 == model
            inds0 = latent_inds0 == model

            ##### indiv Plots ####
            save_filename1 = os.path.join(root, mname + '_' + \
                                    str(n_qp) + 'qps_latents1_comparison_moons.pdf')
            save_filename0 = os.path.join(root, mname + '_' + \
                                     str(n_qp) + 'qps_latents0_comparison_moons.pdf')
            ax = plt.gca()

            orig = ax.scatter(all_latents1[orig_inds1,0],all_latents1[orig_inds1,1],c=color_orig,marker = 'o', alpha=0.1)
            new = ax.scatter(all_latents1[inds1,0],all_latents1[inds1,1],c=model_colors[model-1],marker='o', alpha=0.1)

            plt.legend([orig,new],['original latents 1', mname + ' latents 1'])
            plt.tight_layout()
            plt.axis('square')
            plt.savefig(save_filename1)
            plt.close('all')

            ax = plt.gca()

            orig = ax.scatter(all_latents0[orig_inds0,0],all_latents0[orig_inds0,1],c=color_orig,marker = 'o',alpha=0.1)
            new = ax.scatter(all_latents0[inds0,0],all_latents0[inds0,1],c=model_colors[model-1],marker='o',alpha=0.1)

            plt.legend([orig,new],['original latents 0', mname + ' latents 0'])
            plt.tight_layout()
            plt.axis('square')
            plt.savefig(save_filename0)
            plt.close('all')
            '''
        #### All plots: 0000, 1111
        imdir = root#os.path.join(root, 'lowlowlowlowregcomp')
        orig_inds1 = latent_inds1 == 0
        orig_inds0 = latent_inds0 == 0
        orig_indsa = latent_indsa == 0

        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latents1111_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latents0000_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_0000[orig_inds0,0],all_l_0000[orig_inds0,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_0000[latent_inds0 == 1,0],all_l_0000[latent_inds0 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_0000[latent_inds0 == 2,0],all_l_0000[latent_inds0 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_0000[latent_inds0 == 3,0],all_l_0000[latent_inds0 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_1111[orig_inds1,0],all_l_1111[orig_inds1,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_1111[latent_inds1 == 1,0],all_l_1111[latent_inds1 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_1111[latent_inds1 == 2,0],all_l_1111[latent_inds1 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_1111[latent_inds1 == 3,0],all_l_1111[latent_inds1 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')
        print('Done!')

        #### All plots: 0011, 1100

        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latents1100_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latents0011_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_0011[orig_inds0,0],all_l_0011[orig_inds0,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_0011[latent_inds0 == 1,0],all_l_0011[latent_inds0 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_0011[latent_inds0 == 2,0],all_l_0011[latent_inds0 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_0011[latent_inds0 == 3,0],all_l_0011[latent_inds0 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_1100[orig_inds1,0],all_l_1100[orig_inds1,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_1100[latent_inds1 == 1,0],all_l_1100[latent_inds1 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_1100[latent_inds1 == 2,0],all_l_1100[latent_inds1 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_1100[latent_inds1 == 3,0],all_l_1100[latent_inds1 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')
        print('Done!')

        #### All plots: 1b1b, 0b0b

        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latents1b1b_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latents0b0b_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_0b0b[orig_inds0,0],all_l_0b0b[orig_inds0,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_0b0b[latent_inds0 == 1,0],all_l_0b0b[latent_inds0 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_0b0b[latent_inds0 == 2,0],all_l_0b0b[latent_inds0 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_0b0b[latent_inds0 == 3,0],all_l_0b0b[latent_inds0 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_1b1b[orig_inds1,0],all_l_1b1b[orig_inds1,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_1b1b[latent_inds1 == 1,0],all_l_1b1b[latent_inds1 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_1b1b[latent_inds1 == 2,0],all_l_1b1b[latent_inds1 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_1b1b[latent_inds1 == 3,0],all_l_1b1b[latent_inds1 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')
        print('Done!')

        ## All plots 1b0b, 0b1b
        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latents1b0b_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latents0b1b_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_0b1b[orig_inds0,0],all_l_0b1b[orig_inds0,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_0b1b[latent_inds0 == 1,0],all_l_0b1b[latent_inds0 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_0b1b[latent_inds0 == 2,0],all_l_0b1b[latent_inds0 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_0b1b[latent_inds0 == 3,0],all_l_0b1b[latent_inds0 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_1b0b[orig_inds1,0],all_l_1b0b[orig_inds1,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_1b0b[latent_inds1 == 1,0],all_l_1b0b[latent_inds1 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_1b0b[latent_inds1 == 2,0],all_l_1b0b[latent_inds1 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_1b0b[latent_inds1 == 3,0],all_l_1b0b[latent_inds1 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')
        print('Done!')

        #### All plots: 1b11, 0b00

        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latents1b11_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latents0b00_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_0b00[orig_inds0,0],all_l_0b00[orig_inds0,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_0b00[latent_inds0 == 1,0],all_l_0b00[latent_inds0 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_0b00[latent_inds0 == 2,0],all_l_0b00[latent_inds0 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_0b00[latent_inds0 == 3,0],all_l_0b00[latent_inds0 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_1b11[orig_inds1,0],all_l_1b11[orig_inds1,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_1b11[latent_inds1 == 1,0],all_l_1b11[latent_inds1 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_1b11[latent_inds1 == 2,0],all_l_1b11[latent_inds1 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_1b11[latent_inds1 == 3,0],all_l_1b11[latent_inds1 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')
        print('Done!')

        ## All plots 1b00, 0b11
        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latents1b00_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latents0b11_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_0b11[orig_inds0,0],all_l_0b11[orig_inds0,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_0b11[latent_inds0 == 1,0],all_l_0b11[latent_inds0 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_0b11[latent_inds0 == 2,0],all_l_0b11[latent_inds0 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_0b11[latent_inds0 == 3,0],all_l_0b11[latent_inds0 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_1b00[orig_inds1,0],all_l_1b00[orig_inds1,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_1b00[latent_inds1 == 1,0],all_l_1b00[latent_inds1 == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_1b00[latent_inds1 == 2,0],all_l_1b00[latent_inds1 == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_1b00[latent_inds1 == 3,0],all_l_1b00[latent_inds1 == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')
        print('Done!')





        ## All plots all11, all00
        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latentsa11_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latentsa00_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_a00[orig_indsa,0],all_l_a00[orig_indsa,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_a00[latent_indsa == 1,0],all_l_a00[latent_indsa == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_a00[latent_indsa == 2,0],all_l_a00[latent_indsa == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_a00[latent_indsa == 3,0],all_l_a00[latent_indsa == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_a11[orig_indsa,0],all_l_a11[orig_indsa,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_a11[latent_indsa == 1,0],all_l_a11[latent_indsa == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_a11[latent_indsa == 2,0],all_l_a11[latent_indsa == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_a11[latent_indsa == 3,0],all_l_a11[latent_indsa == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')


        ## All plots all1b, all0b
        save_filename1 = os.path.join(imdir,  \
                                 str(n_qp) + '_qps_moons_latentsa1b_comparison_all.pdf')
        save_filename0 = os.path.join(imdir, \
                                 str(n_qp) + '_qps_moons_latentsa0b_comparison_all.pdf')

        ax = plt.gca()

        orig = ax.scatter(all_l_a0b[orig_indsa,0],all_l_a0b[orig_indsa,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_a0b[latent_indsa == 1,0],all_l_a0b[latent_indsa == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_a0b[latent_indsa == 2,0],all_l_a0b[latent_indsa == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_a0b[latent_indsa == 3,0],all_l_a0b[latent_indsa == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 0', mnames[0],mnames[1],mnames[2]])#,mnames[3]])

        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename0)
        plt.close('all')
        print('Done!')

        ax = plt.gca()

        orig = ax.scatter(all_l_a1b[orig_indsa,0],all_l_a1b[orig_indsa,1],\
                            c=color_orig,marker = 'o',alpha=0.1)
        van = ax.scatter(all_l_a1b[latent_indsa == 1,0],all_l_a1b[latent_indsa == 1,1],\
                            c=model_colors[0],marker='o',alpha=0.1)
        beta = ax.scatter(all_l_a1b[latent_indsa == 2,0],all_l_a1b[latent_indsa == 2,1],\
                            c=model_colors[1],marker='o',alpha=0.1)
        #vq = ax.scatter(all_latents1[latent_inds1 == 3,0],all_latents1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
        qp = ax.scatter(all_l_a1b[latent_indsa == 3,0],all_l_a1b[latent_indsa == 3,1],\
                            c=model_colors[2],marker='o',alpha=0.1)

        plt.legend([orig,van,beta,qp],['Original Moon 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename1)
        plt.close('all')
        print('Done!')
