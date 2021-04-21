from qp_vae_bird import QP_VAE
from b_vae_bird import B_VAE
from vae_bird import VAE
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
from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE, VAE
from ava.models.vae_dataset import get_syllable_partition, \
	get_syllable_data_loaders
from ava.preprocessing.preprocess import process_sylls, \
	tune_syll_preprocessing_params
from ava.preprocessing.utils import get_spec
from ava.segmenting.refine_segments import refine_segments_pre_vae
from ava.segmenting.segment import tune_segmenting_params, segment
from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from qpvae_utils import get_qps_km, qpvae_ds, calc_dkl


def bird_model_script(n_train_runs = 10, root = '',figs=False):

########## Setting up directory lists: separate into train, test dirs
############# Expected File Structure

# Root (code, stuff, etc)
# |
# | --- Animal 1
# |      |
# |      | - day 1
# |      |   | - data dirs
# |      |
# |      | - day 2
# |          | - data dirs
# |
# | --- Animal 2
# | ...

    datadir = '/home/mrmiews/Desktop/Pearson_Lab/bird_data'
    #root = '/home'
    trainDays = ['80','100','115']
    ds1 = ['blu_258']
    ds2 = ['blu_288']
    dsb = ['blu_258', 'blu_288']
    z_dim = 32

    ds1_audio_dirs = [os.path.join(datadir,animal,'audio',day) for animal in ds1 for day in trainDays]
    ds2_audio_dirs = [os.path.join(datadir,animal,'audio',day) for animal in ds2 for day in trainDays]
    dsb_audio_dirs = [os.path.join(datadir,animal,'audio',day) for animal in dsb for day in trainDays]

    ds1_segment_dirs = [os.path.join(datadir,animal,'segs',day) for animal in ds1 for day in trainDays]
    ds2_segment_dirs = [os.path.join(datadir,animal,'segs',day) for animal in ds2 for day in trainDays]
    dsb_segment_dirs = [os.path.join(datadir,animal,'segs',day) for animal in dsb for day in trainDays]

    ds1_spec_dirs = [os.path.join(datadir,animal,'h5s',day) for animal in ds1 for day in trainDays]
    ds2_spec_dirs = [os.path.join(datadir,animal,'h5s',day) for animal in ds2 for day in trainDays]
    dsb_spec_dirs = [os.path.join(datadir,animal,'h5s',day) for animal in dsb for day in trainDays]

    ds1_proj_dirs = [os.path.join(datadir,animal,'proj',day) for animal in ds1 for day in trainDays]
    ds2_proj_dirs = [os.path.join(datadir,animal,'proj',day) for animal in ds2 for day in trainDays]
    dsb_proj_dirs = [os.path.join(datadir,animal,'proj',day) for animal in dsb for day in trainDays]


    model_filename = os.path.join(root,'joint_template','checkpoint_200.tar')
    plots_dir = os.path.join(root,'plots')

    ds1_dc = DataContainer(projection_dirs=ds1_proj_dirs, audio_dirs=ds1_audio_dirs, \
        segment_dirs=ds1_segment_dirs,plots_dir=plots_dir, model_filename=model_filename)
    ds2_dc = DataContainer(projection_dirs=ds2_proj_dirs, audio_dirs=ds2_audio_dirs, \
        segment_dirs=ds2_segment_dirs, plots_dir=plots_dir, model_filename=model_filename)
    dsb_dc = DataContainer(projection_dirs=dsb_proj_dirs, audio_dirs=dsb_audio_dirs,\
        segment_dirs=dsb_segment_dirs, plots_dir=plots_dir,model_filename=model_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ####################################
    # 0.5) Define segmenting parameters #
    #####################################

    segment_params = {
        'min_freq': 10, # minimum frequency
        'max_freq': 25e3, #maximum frequency
        'nperseg': 512, # FFT, length of Hann window
        'noverlap': 256, # FFT, overlap in sequences
        'spec_min_val': 2.0, # minimum log-spectrogram value
        'spec_max_val': 6.0, # maximum log-spectrogram value
        'fs': 44100, # audio samplerate
        'get_spec': get_spec, # figure out what this is
        'min_dur': 0.015, # minimum syllable duration
        'max_dur': 0.2, #maximum syllable duration
        'smoothing_timescale': 0.007, #amplitude
        'temperature': 0.5, # softmax temperature parameter
        'softmax': False, # apply softmax to frequency bins to calculate amplitude
        'th_1': 1.5, # segmenting threshold 1
        'th_2': 2.0, # segmenting threshold 2
        'th_3': 2.5, # segmenting threshold 3
        'window_length': 0.12, # FFT window??
        'algorithm': get_onsets_offsets, #finding syllables
        'num_freq_bins': X_SHAPE[0],
        'num_time_bins': X_SHAPE[1],
        'mel': True, # Frequency spacing: mel-spacing for birbs
        'time_stretch': True, #are we warping time?
        'within_syll_normalize': False, #normalize within syllables?
        'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
            'spec_max_val'), #I'm sure this does something
        'int_preprocess_params': tuple([]), #i'm ALSO sure this does something
        'binary_preprocess_params': ('mel', 'within_syll_normalize'), #shrug
        }

    segment_params = tune_segmenting_params(ds1_audio_dirs, segment_params)

#############################
# 1) Amplitude segmentation #
#############################


    for audio_dir, segment_dir in zip(ds1_audio_dirs, ds1_segment_dirs):
        if os.path.isdir(segment_dir):
            if len(os.listdir(segment_dir)) == 0: #if segmented files don't already exist
                segment(audio_dir, segment_dir, segment_params)
            else: continue
        else: segment(audio_dir, segment_dir, segment_params)

    for audio_dir, segment_dir in zip(ds2_audio_dirs, ds2_segment_dirs):
        if os.path.isdir(segment_dir):
            if len(os.listdir(segment_dir)) == 0: #if segmented files don't already exist
                segment(audio_dir, segment_dir, segment_params)
            else: continue
        else: segment(audio_dir, segment_dir, segment_params)

    for audio_dir, segment_dir in zip(dsb_audio_dirs, dsb_segment_dirs):
        if os.path.isdir(segment_dir):
            if len(os.listdir(segment_dir)) == 0: #if segmented files don't already exist
                segment(audio_dir, segment_dir, segment_params)
            else: continue
        else: segment(audio_dir, segment_dir, segment_params)


#######################################
# 1.75(?)) Preprocessing spectrograms #
# Necessary for coloring latent plots!
#######################################


    preprocess_params = copy.copy(segment_params)

    preprocess_params["mel"] = False
    preprocess_params["time_stretch"] = True
    preprocess_params["'real_preprocess_params'"] = ('min_freq', 'max_freq', 'spec_min_val', \
            'spec_max_val', 'max_dur')
    preprocess_params["int_preprocess_params"] = ('nperseg','noverlap')
    preprocess_params["binary_preprocess_params"] = ('time_stretch','mel','within_syll_normalize')
    preprocess_params["sylls_per_file"] = 20
    preprocess_params["max_num_syllables"] = None

    preprocess_params = tune_syll_preprocessing_params(ds1_audio_dirs, ds1_segment_dirs, \
            preprocess_params)

    #gen_d1 = zip(ds1_audio_dirs,ds1_segment_dirs,ds1_spec_dirs, repeat(preprocess_params))
    #gen_d2 = zip(ds2_audio_dirs,ds2_segment_dirs,ds2_spec_dirs, repeat(preprocess_params))
    gen_db = zip(dsb_audio_dirs,dsb_segment_dirs,dsb_spec_dirs, repeat(preprocess_params))

    Parallel(n_jobs=4)(delayed(process_sylls)(*args) for args in gen_db)
    #Parallel(n_jobs=4)(delayed(process_sylls)(*args) for args in gen_test)

    split = 0.6
    partition_ds1 = get_syllable_partition(ds1_spec_dirs, split)
    partition_ds2 = get_syllable_partition(ds2_spec_dirs, split)
    partition_dsb = get_syllable_partition(dsb_spec_dirs,split)

    #latent_p_ds1 = get_syllable_partition(ds1_spec_dirs, split)
    #latent_p_ds2 = get_syllable_partition(ds2_spec_dirs, split)
    #latent_p_dsb = get_syllable_partition(dsb_spec_dirs,split)
    #latent_p_ds1['test'] = latent_p_ds1['train']
    #latent_p_ds2['test'] = latent_p_ds2['train']
    #latent_p_dsb['test'] = latent_p_dsb['train']


    num_workers = min(7, os.cpu_count()-1)
    ###### Training models only ever see ds2 ###########
    seq_loaders_ds1 = get_syllable_data_loaders(partition_ds1, \
                    num_workers=num_workers, batch_size=256)
    seq_loaders_ds2 = get_syllable_data_loaders(partition_ds2, \
                    num_workers=num_workers, batch_size=256)
    ######## Only Template sees this ########
    joint_loaders_dsb = get_syllable_data_loaders(partition_dsb, \
                    num_workers=num_workers, batch_size=256)

    ######## all models can see these, but none see for training #######
    lat_load_ds1 = get_syllable_data_loaders(partition_ds1, \
                    num_workers=num_workers, batch_size=256,shuffle=(False,True))
    lat_load_ds2 = get_syllable_data_loaders(partition_ds2, \
                    num_workers=num_workers, batch_size=256,shuffle=(False,True))
    lat_load_dsb = get_syllable_data_loaders(partition_dsb, \
                    num_workers=num_workers, batch_size=256,shuffle=(False,True))

    vae_template_seq_save = os.path.join(root,'vae_mnist_seq_template')
    vae_template_joint_save = os.path.join(root,'vae_mnist_joint_template')
    joint_template_fname = os.path.join(vae_template_joint_save, 'checkpoint_200.tar')
    seq_template_fname = os.path.join(vae_template_seq_save, 'checkpoint_200.tar')

    vae_template_joint = VAE(save_dir=vae_template_joint_save, z_dim=z_dim)
    vae_template_seq = VAE(save_dir=vae_template_seq_save, z_dim=z_dim)

    print('Training joint template VAE')

    if not os.path.isfile(joint_template_fname):
        vae_template_joint.train_loop(joint_loaders_dsb,epochs=201,test_freq=None,vis_freq=5)
    else:
        vae_template_joint.load_state(joint_template_fname)

    latents_1j = vae_template_joint.get_latent(lat_load_ds1['train'])
    latents_2j = vae_template_joint.get_latent(lat_load_ds2['train'])
    latents_aj = vae_template_joint.get_latent(lat_load_dsb['train'])

    # then sequential
    vae_template_seq_save = os.path.join(root,'vae_mnist_seq_template')
    vae_template_seq = VAE(save_dir = vae_template_seq_save, z_dim = z_dim)
    seq_template_fname = os.path.join(vae_template_seq_save, 'checkpoint_200.tar')

    print('Training sequential template VAE')
    if not os.path.isfile(seq_template_fname):
        vae_template_seq.train_loop(seq_loaders_ds1,epochs=201,test_freq=None,vis_freq=5)
    else:
        vae_template_seq.load_state(seq_template_fname)

    latents_1s = vae_template_seq.get_latent(lat_load_ds1['train'])
    latents_2s = vae_template_seq.get_latent(lat_load_ds2['train'])
    latents_as = vae_template_seq.get_latent(lat_load_dsb['train'])
    print(latents_1j.shape)
    n_qps = [32,64]#[64,128]
    n_train1 = len(latents_1j)
    print(n_train1)

    n_train2 = len(latents_2j)
    #n_qps.append(n_train1)

    old_datax1j = []
    old_dataz1j = []
    old_datax1s = []
    old_dataz1s = []

    xj_filenames = []
    xs_filenames = []

    nfiles_s = []
    nfiles_j = []

    qp_folder = os.path.join(root, 'anchor_points')
    if not os.path.isdir(qp_folder):
        os.mkdir(qp_folder)

    spec_filef = os.path.join(datadir,'anchor_point_specs')
    if not os.path.isdir(spec_filef):
        os.mkdir(spec_filef)

    for n_qp in n_qps:

        qp_picklej = os.path.join(qp_folder,'anchors_joint_' + str(n_qp) + '.pickle')
        qp_pickles = os.path.join(qp_folder,'anchors_seq_' + str(n_qp) + '.pickle')
        nfs = 0
        nfj = 0
        #first, check if we have already saved qps & files for this n qps

        if not os.path.isfile(qp_picklej):
            print('finding joint centroids')

            oldxj_tmp = []
            oldzj_tmp = []
            tmp_fnj = []

            if n_qp < n_train1:
                kmj = KMeans(n_clusters=n_qp).fit(latents_1j)
                joint_cent = kmj.cluster_centers_
                for centroid_num, centroid in enumerate(joint_cent):
                    cdj = np.linalg.norm(centroid.T - latents_1j, axis=1)
                    min_dj = np.argmin(cdj)
                    tmp_xj = lat_load_ds1['train'].dataset[min_dj]
                    tmp_zj = latents_1j[min_dj,:]
                    oldxj_tmp.append(tmp_xj)
                    oldzj_tmp.append(torch.from_numpy(tmp_zj).type(torch.FloatTensor))
                oldzj_tmp = torch.stack(oldzj_tmp)

            else:
                oldxj_tmp = lat_load_ds1['train'].dataset[list(range(len(lat_load_ds1['train'].dataset)))]
                oldzj_tmp = torch.from_numpy(latents_1j).type(torch.FloatTensor)

            old_datax1j.append(torch.stack(oldxj_tmp))
            old_dataz1j.append(oldzj_tmp)

            n_cent = len(oldxj_tmp)
            write_num = 0
            nper = preprocess_params['sylls_per_file']
            while n_cent >= nper:
                save_fnj = os.path.join(qp_folder,'anchors_joint_' + str(n_qp) + '_' + str(write_num) + '.hdf5')
                tmp_fnj.append(save_fnj)
                xj_save = oldxj_tmp[write_num*nper:(write_num + 1)*nper]
                xj_save = torch.stack(xj_save).detach().cpu().numpy()
                with h5py.File(save_fnj,'w') as f:
                    f.create_dataset('specs',data=xj_save)
                    f.create_dataset('lengt',data=[nper])
                write_num += 1
                n_cent -= nper
                nfj += nper

            if n_cent > 0:
                save_fnj = os.path.join(qp_folder,'anchors_joint_' + str(n_qp) + '_' + str(write_num) + '.hdf5')
                tmp_fnj.append(save_fnj)
                xj_save = oldxj_tmp[write_num*nper::]
                xj_save = torch.stack(xj_save).detach().cpu().numpy()
                with h5py.File(save_fnj,'w') as f:
                    f.create_dataset('specs',data=xj_save)
                    f.create_dataset('lengt', data=[xj_save.shape[0]])

                nfj += xj_save.shape[0]

            model_dat = {'x':oldxj_tmp,'z':oldzj_tmp,'fns':tmp_fnj, 'nf':nfj}
            with open(qp_picklej,'wb') as f:
                pickle.dump(model_dat,f,protocol=pickle.HIGHEST_PROTOCOL)

            xj_filenames.append(tmp_fnj)
            nfiles_j.append(nfj)
        else:
            with open(qp_picklej,'rb') as f:
                data = pickle.load(f)
            old_datax1j.append(torch.stack(data['x']))
            old_dataz1j.append(data['z'])
            xj_filenames.append(data['fns'])
            nfiles_j.append(data['nf'])

####### do the above but for sequential training
        if not os.path.isfile(qp_pickles):
            print('finding sequential centroids')
            oldxs_tmp = []
            oldzs_tmp = []
            tmp_fns = []

            if n_qp < n_train1:
                ###### Find cluster centers #####
                kms = KMeans(n_clusters=n_qp).fit(latents_1s)
                seq_cent = kms.cluster_centers_
                for centroid_num, centroid in enumerate(seq_cent):
                    ##### Find closest latents, corr. x #####
                    cds = np.linalg.norm(centroid.T - latents_1s, axis=1)
                    min_ds = np.argmin(cds)
                    tmp_xs = lat_load_ds1['train'].dataset[min_ds] #### fix this: need to get from dataset instead of dataloader
                    tmp_zs = latents_1s[min_ds,:]
                    oldxs_tmp.append(tmp_xs)
                    oldzs_tmp.append(torch.from_numpy(tmp_zs).type(torch.FloatTensor))
                oldzs_tmp = torch.stack(oldzs_tmp)

            else:
                #### If we are using all the data, just grab the data + latents, dont cluster
                oldxs_tmp = lat_load_ds1['train'].dataset[list(range(len(lat_load_ds1['train'].dataset)))]
                oldzs_tmp = torch.from_numpy(latents_1s).type(torch.FloatTensor)

            old_datax1s.append(torch.stack(oldxs_tmp))
            old_dataz1s.append(oldzs_tmp)

            n_cent = len(oldxs_tmp)
            write_num = 0
            nper = preprocess_params['sylls_per_file']
            while n_cent >= nper:
                save_fns = os.path.join(qp_folder,'anchors_seq_' + str(n_qp) + '_' + str(write_num) + '.hdf5')
                tmp_fns.append(save_fns)
                xs_save = oldxs_tmp[write_num*nper:(write_num + 1)*nper]
                xs_save = torch.stack(xs_save).detach().cpu().numpy()
                with h5py.File(save_fns,'w') as f:
                    f.create_dataset('specs',data=xs_save)
                    f.create_dataset('lengt', data=nper)
                write_num += 1
                n_cent -= nper
                nfs += nper

            if n_cent > 0:
                save_fns = os.path.join(qp_folder,'anchors_seq_' + str(n_qp) + '_' + str(write_num) + '.hdf5')
                tmp_fns.append(save_fns)
                xs_save = oldxs_tmp[write_num*nper::]
                xs_save = torch.stack(xs_save).detach().cpu().numpy()
                with h5py.File(save_fns,'w') as f:
                    f.create_dataset('specs',data=xs_save)
                    f.create_dataset('lengt', data=[xs_save.shape[0]])
                    nfs += xs_save.shape[0]

            model_dat = {'x':oldxs_tmp,'z':oldzs_tmp,'fns':tmp_fns,'nf':nfs}
            with open(qp_pickles,'wb') as f:
                pickle.dump(model_dat,f,protocol=pickle.HIGHEST_PROTOCOL)

        #    print(tmp_fns)
            nfiles_s.append(nfs)
            xs_filenames.append(tmp_fns)
        else:
            with open(qp_pickles,'rb') as f:
                data = pickle.load(f)
            old_datax1s.append(torch.stack(data['x']))
            old_dataz1s.append(data['z'])
            xs_filenames.append(data['fns'])
            nfiles_s.append(data['nf'])

    ############# Create new datasets for training ##########
    qp_ds_copies = []
    #print(xs_filenames)
    for qp_ind, n_qp in enumerate(n_qps):
        #print(qp_ind)
        #print(xs_filenames[0])
        #### switch copy.copy to copy.deepcopy
        oldfn_s = xs_filenames[qp_ind]
        oldfn_j = xj_filenames[qp_ind]
        print(len(seq_loaders_ds2['train'].dataset))
        #if qp_ind > 0:
            #print(seq_loaders_ds2['train'].dataset.filenames)
        ds2s_dict_copy = copy.copy(seq_loaders_ds2)
        ds2j_dict_copy = copy.copy(seq_loaders_ds2)

        ds2s_dl = copy.copy(ds2s_dict_copy['train'])
        #print(type(ds2s_dl.dataset.filenames))
        ds2j_dl = copy.copy(ds2j_dict_copy['train'])


        ds2s_ds = copy.copy(ds2s_dl.dataset)
        #print(type(ds2s_ds.filenames))
        ds2j_ds = copy.copy(ds2j_dl.dataset)

        fn_s_tmp = copy.copy(ds2s_ds.filenames)
        fn_j_tmp = copy.copy(ds2j_ds.filenames)

        fn_s_tmp.extend(oldfn_s)
        fn_j_tmp.extend(oldfn_j)

        ds2s_ds.filenames = fn_s_tmp
        ds2s_ds.lengt += nfiles_s[qp_ind]
        #print(oldfn_s)
        #print(type(ds2s_ds.filenames))
        ds2j_ds.filenames = fn_j_tmp
        ds2j_ds.lengt += nfiles_j[qp_ind]
        #print(ds2j_ds.filenames)
        j_loader_tmp = DataLoader(ds2j_ds, num_workers=num_workers, \
                shuffle=True, batch_size=256)
        s_loader_tmp = DataLoader(ds2s_ds, num_workers=num_workers, \
                shuffle=True, batch_size=256)

        print(len(ds2j_ds))
        ds2s_dict_copy['train'] = j_loader_tmp#ds2s_ds.filenames
        ds2j_dict_copy['train'] = s_loader_tmp#ds2j_ds.filenames
        print(len(ds2s_dict_copy['train'].dataset))

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
    batch_size = 256
    for run_ind in range(n_train_runs):
        print('Beginning Training Run ', run_ind + 1)

        for qp_ind, n_qp in enumerate(n_qps):

            print('N QPs: ', n_qp)
            beta_qpvae = 2.75*batch_size/n_qp

            oldx1j = old_datax1j[qp_ind]
            oldx1s = old_datax1s[qp_ind]

            #oldy1j = old_datay1j[qp_ind]
            #oldy1s = old_datay1s[qp_ind]

            oldz1j = old_dataz1j[qp_ind]
            oldz1s = old_dataz1s[qp_ind]

            ds_dictj,ds_dicts = qp_ds_copies[qp_ind]
            print(ds_dictj['train'].dataset.filenames[-4::])
            print(len(ds_dictj['train'].dataset))

            vae_1j_anchor_save = os.path.join(root,'vae1j_bird_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            vae_1s_anchor_save = os.path.join(root,'vae1s_bird_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            bvae_1j_anchor_save = os.path.join(root,'bvae1j_bird_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            bvae_1s_anchor_save = os.path.join(root,'bvae1s_bird_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            qpvae_1j_anchor_save = os.path.join(root,'qpvae1j_bird_run_'+str(run_ind) + '_nqp_' + str(n_qp))
            qpvae_1s_anchor_save = os.path.join(root,'qpvae1s_bird_run_'+str(run_ind) + '_nqp_' + str(n_qp))

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
                qpvae1j_anchor.train_loop(seq_loaders_ds2,oldx1j, oldz1j,epochs=201,test_freq=None,vis_freq=10)
            else:
                qpvae1j_anchor.load_state(qpvae_1j_anchor_fn)
            if not os.path.isfile(qpvae_1s_anchor_fn):
                qpvae1s_anchor.train_loop(seq_loaders_ds2,oldx1s, oldz1s, epochs=201,test_freq=None,vis_freq=10)
            else:
                qpvae1s_anchor.load_state(qpvae_1s_anchor_fn)


            ############### Get Latents ###########
            vae1j_anchor_la = vae1j_anchor.get_latent(lat_load_dsb['train'])
            vae1s_anchor_la = vae1s_anchor.get_latent(lat_load_dsb['train'])

            bvae1j_anchor_la = bvae1j_anchor.get_latent(lat_load_dsb['train'])
            bvae1s_anchor_la = bvae1s_anchor.get_latent(lat_load_dsb['train'])

            qpvae1j_anchor_la = qpvae1j_anchor.get_latent(lat_load_dsb['train'])
            qpvae1s_anchor_la = qpvae1s_anchor.get_latent(lat_load_dsb['train'])

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

            print('Current Rosetta Joint Distortion:,', np.mean(mse_matj[:,run_ind,qp_ind,2]))
            print('Current VAE Joint Distortion:,', np.mean(mse_matj[:,run_ind,qp_ind,0]))
            print('Current BVAE Joint Distortion:,', np.mean(mse_matj[:,run_ind,qp_ind,1]))

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
