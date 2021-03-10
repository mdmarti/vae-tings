#### TOY MODEL CODE ######


#goal train VAEs on subsets of data. we want to compare between vanilla VAE, VQ-VAE, and our method

#specifically, this is with the mnist dataset
# we will train on a subset of the digits, then on a random subset of the data across digits

#umap latent space for each model, compare all of them


###### Imports ################
#from ava.models.vae_MNIST import VAE
#from ava.models.b_vae_MNIST import B_VAE
#from ava.models.qp_vae_MNIST import QP_VAE
#from ava.models.vq_vae_MNIST import VQ_VAE
#from ava.plotting import grid_plot
from vq_vae_MNIST import VQ_VAE
from qp_vae_MNIST import QP_VAE
from b_vae_MNIST import B_VAE
from vae_MNIST import VAE
import matplotlib.pyplot as plt

import numpy as np
import os
import copy
import h5py
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
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



##################################
# Definitions building on dataset, dataloader for my own clarity
##################################

class MNISTDataset(Dataset):
    """torch.utils.data.Dataset for MNIST images"""

    def __init__(self, data, transform=None):
        """
        Create a torch.utils.data.Dataset for MNIST images.

        Parameters
        ----------
        filenames : list of strings
	       List of hdf5 files containing syllable spectrograms.
        sylls_per_file : int
            Number of syllables in each hdf5 file.
        transform : None or function, optional
            Transformation to apply to each item. Defaults to None (no
            transformation)
        """
        (self.images, self.labels) = data
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        result = []
        label = []
        single_index = False
        try:
            iterator = iter(index)
        except TypeError:
            index = [index]
            single_index = True
        for i in index:

            tmp_im = self.images[index,:,:]

            if self.transform:
                tmp_im = self.transform(tmp_im)

            result.append(tmp_im)
            label.append(self.labels[index,])
        if single_index:
            return (result[0], label[0])
        return (result, label)

def calc_dkl(model1, model2, dataLoader):

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    kl = np.zeros([len(dataLoader),])
    for batch_idx, data in enumerate(dataLoader):

        (images, labels) = data
        images = images.to(device)
        with torch.no_grad():
            mu1,u1,d1 = model1.encode(images)
            mu2,u2,d2 = model2.encode(images)

            latent_dist1 = LowRankMultivariateNormal(mu1, u1, d1)
            latent_dist2 = LowRankMultivariateNormal(mu2, u2, d2)

            lp1 = latent_dist1.log_prob(mu1).detach().cpu().numpy()
            lp2 = latent_dist2.log_prob(mu2).detach().cpu().numpy()

            kl[batch_idx] = np.sum(np.multiply(np.exp(lp1), (lp1 - lp2)))

    return np.sum(kl)
##################################
# Import MNIST
##################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose training options')
    parser.add_argument("-qp", \
            help='Chooses quantization procedure for initial model. If qp=="vq", quantaziation done by vqvae. If qp=="km", done by KMeans.',\
            action='store',required=True,dest='quant_type')
    parser.add_argument("-analysis", "-a", \
            help='Chooses analysis. Takes three values. "models", "betas", or "both". If model, compares between models, if betas, compares qpvaes between betas, if both does both.',\
            action='store',required=True,dest='analysis_type')

    vals = parser.parse_args()
    vals = vars(vals)
    atype = vals['analysis_type']
    qtype = vals['quant_type']

    model_compare = (atype == 'models' or atype == 'both')
    betas_qps_compare = (atype == 'betas' or atype == 'both')

#    root = '/home/mmartinez/autoencoded-vocal-analysis/ava_qpe_experiments/actual_experiments_ratio'
    root = '/home/mrmiews/Desktop/Pearson_Lab/models'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist = tf.keras.datasets.mnist
    #loads these in as np arrays- much better than torch smh
    (x_train,y_train), (x_test,y_test) = mnist.load_data()

    x_all = np.vstack((x_train,x_test)).astype(np.float32)/255
    y_all = np.hstack((y_train,y_test)).astype(np.int)
    #print(x_all.shape)

    unique_labels = np.unique(y_all)

##################################
# Split MNIST into random subsets (across digits)
# randomly select 65% of data, then a separate quarter
# The end product of this should be a torch dataloader
# with the required train, test splits. maybe use a custom class
# building on torch DataLoader?
##################################
# 0.5 train/test split

    # this will actually be randomly splitting to disjoint groups
    # one set gets some digits, another gets the others

    batch_size = 128
    num_workers = min(7,os.cpu_count()-1)

    split = 0.6

    #code for setting up between-digit split (disjoint training sets across digits)
    dig1 = np.random.choice(unique_labels,5, replace = False)
    dig2 = np.array([dig for dig in unique_labels if dig not in dig1])

    x1 = np.array([x_data for ind, x_data in enumerate(x_all) if y_all[ind] in dig1])
    #print(x1.shape)
    x2 = np.array([x_data for ind, x_data in enumerate(x_all) if y_all[ind] in dig2])
    y1 = np.hstack([y for y in y_all if y in dig1])
    y2 = np.hstack([y for y in y_all if y in dig2])

    n_dat1 = len(y1)
    n_dat2 = len(y2)
    if not(n_dat1 == n_dat2):
        print('number of numbers not equal!!!!')
        print('you should fix this somehow')


    n_train1 = int(round(split * n_dat1))
    n_train2 = int(round(split * n_dat2))
    n_test1 = n_dat1 - n_train1
    n_test2 = n_dat2 - n_train2

    np.random.seed(35)

    order1 = np.random.permutation(n_dat1)
    order2 = np.random.permutation(n_dat2)


    train1 = order1[0:n_train1]
    train2 = order2[0:n_train2]

    test1 = order1[n_train1::]
    test2 = order2[n_train2::]

    x_train1 = x1[train1,:,:]
    y_train1 = y1[train1,]

    x_test1 = x1[test1,:,:]
    y_test1 = y1[test1,]

    x_train2 = x2[train2,:,:]
    y_train2 = y2[train2,]

    x_test2 = x2[test2,:,:]
    y_test2 = y2[test2,]

    train_dataset1 = MNISTDataset((x_train1,y_train1))
    test_dataset1 = MNISTDataset((x_test1,y_test1))

    train_across_dataloader1 = DataLoader(train_dataset1, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)
    latents_dataloader1 = DataLoader(train_dataset1, batch_size = batch_size, \
        shuffle = False, num_workers=num_workers)
    test_across_dataloader1 = DataLoader(test_dataset1, batch_size = batch_size, \
        shuffle = False, num_workers=num_workers)
    #print(len(train_across_dataloader1.dataset))
    across1 = {'train':train_across_dataloader1, 'test':test_across_dataloader1}

    train_dataset2 = MNISTDataset((x_train2,y_train2))
    test_dataset2 = MNISTDataset((x_test2,y_test2))

    train_across_dataloader2 = DataLoader(train_dataset2, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)
    test_across_dataloader2 = DataLoader(test_dataset2, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)
    latents_dataloader2 = DataLoader(train_dataset2, batch_size = batch_size, \
        shuffle = False, num_workers=num_workers)

    across2 = {'train': train_across_dataloader2, 'test': test_across_dataloader2}

    data_across_dig = {
        'train_dat1': (x_train1,y_train1),
        'train_dat2': (x_train2,y_train2),
        'test_dat1': (x_test1,y_test1),
        'test_dat2': (x_test2,y_test2)
        }

######################
# Great! we have split our data into two sets: train and test
# now we want to initialize our models.
# We want two VQ-VAEs, two vanilla VAEs, two of our version
# let's just do 2 at a time to minimize worklaod
######################
    n_qps = [16,32,64,128]#[64,128]

    z_dim = 32

    beta_bvae = 5

    ####### Check that model architectures are same (except for VQVAE, that's deterministic)
    print(atype)
    print()


    ######## Training template VAEs #######
    if qtype == 'km':
        vae_template1_save = os.path.join(root, 'vae_MNIST_template1')
        vae_template2_save = os.path.join(root,'vae_MNIST_template2')
        vae_template1 = VAE(save_dir = vae_template1_save, z_dim = z_dim)
        vae_template2 = VAE(save_dir = vae_template2_save, z_dim = z_dim)
        template1_fname = os.path.join(vae_template1_save, 'checkpoint_200.tar')
        template2_fname = os.path.join(vae_template2_save, 'checkpoint_200.tar')
        print('Training template VAEs')
        if not os.path.isfile(template1_fname):
            vae_template1.train_loop(across1,epochs=201,test_freq=None,vis_freq=5)
        else:
            vae_template1.load_state(template1_fname)
        if not os.path.isfile(template2_fname):
            vae_template2.train_loop(across1,epochs=201,test_freq=None,vis_freq=5)
        else:
            vae_template2.load_state(template2_fname)
    elif atype == 'qp':
        vqvae_template1_save = os.path.join(root, 'vqvae_MNIST_template1')
        vqvae_template2_save = os.path.join(root,'vqvae_MNIST_template2')
        vqvae_template1 = VQ_VAE(save_dir = vae_template1_save, z_dim = z_dim,beta=1.25,n_embeddings=n_qp)
        vqvae_template2 = VQ_VAE(save_dir = vae_template2_save, z_dim = z_dim,beta=1.25)
        template1_fname = os.path.join(vae_template1_save, 'checkpoint_200.tar')
        template2_fname = os.path.join(vae_template2_save, 'checkpoint_200.tar')

    latents1 = vae_template1.get_latent(latents_dataloader1)
    latents2 = vae_template2.get_latent(latents_dataloader2)

    qps_l1 = []
    qps_l2 = []
    old_datax1 = []
    old_datax2 = []
    old_datay1 = []
    old_datay2 = []
    old_dataz1 = []
    old_dataz2 = []

    ######## Making a folder to store QPs ########
    qp_folder = os.path.join(root, 'qps_from_models')
    if not os.path.isdir(qp_folder):
        os.mkdir(qp_folder)


    ######## Clustering to find QPs OR loading existing qps ########
    for n_qp in n_qps:

        if qtype == 'km':
            qp_filename1 = os.path.join(qp_folder,'qps_model1_' + str(n_qp) + '_qps.pickle')
            qp_filename2 = os.path.join(qp_folder,'qps_model2_' + str(n_qp) + '_qps.pickle')
            qp1_tmp = []
            qp2_tmp = []
            if os.path.isfile(qp_filename1):
                print('Loading QPs for model 1, n QP = ', n_qp)
                with open(qp_filename1,'rb') as f:
                    data = pickle.load(f)
                old_datax1.append(torch.from_numpy(data['x1']).type(torch.FloatTensor))
                old_datay1.append(data['y1'])
                old_dataz1.append(torch.from_numpy(data['z1']).type(torch.FloatTensor))

            else:
                print('Finding QPs for model 1, n QP = ', n_qp)
                qp1_lat = KMeans(n_clusters=n_qp).fit(latents1).cluster_centers_

                for centroid_n,qp in enumerate(qp1_lat):
                    cent_dists = np.linalg.norm(qp - latents1, axis=1);
                    min_dist = np.argmin(cent_dists)
                    qp1_tmp.append(min_dist)


                x1tmp = x_train1[qp1_tmp,:,:]
                y1tmp = y_train1[qp1_tmp,]
                z1tmp = latents1[qp1_tmp,:]

                model1dat = {'x1':x1tmp, 'y1':y1tmp, 'z1': z1tmp}
                old_datax1.append(torch.from_numpy(x1tmp).type(torch.FloatTensor))
                old_datay1.append(y1tmp)
                old_dataz1.append(torch.from_numpy(z1tmp).type(torch.FloatTensor))

                with open(qp_filename1,'wb') as f:
                    pickle.dump(model1dat,f,protocol=pickle.HIGHEST_PROTOCOL)

                qps_l1.append(qp1_lat)
                print('Done with model 1')

            if os.path.isfile(qp_filename2):
                print('Loading QPs for model 2, n QP = ', n_qp)
                with open(qp_filename2,'rb') as f:
                    data = pickle.load(f)
                old_datax2.append(torch.from_numpy(data['x2']).type(torch.FloatTensor))
                old_datay2.append(data['y2'])
                old_dataz2.append(torch.from_numpy(data['z2']).type(torch.FloatTensor))

            else:
                print('Finding QPs for model 2, n QP = ', n_qp)

                qp2_lat = KMeans(n_clusters=n_qp).fit(latents2).cluster_centers_

                for centroid_n,qp in enumerate(qp2_lat):
                    cent_dists = np.linalg.norm(qp - latents2, axis=1);
                    min_dist = np.argmin(cent_dists)
                    qp2_tmp.append(min_dist)

                x2tmp = x_train2[qp2_tmp,:,:]
                y2tmp = y_train2[qp2_tmp,]
                z2tmp = latents2[qp2_tmp,:]

                model2dat = {'x2':x2tmp, 'y2':y2tmp, 'z2': z2tmp}
                old_datax2.append(torch.from_numpy(x2tmp).type(torch.FloatTensor))
                old_datay2.append(y2tmp)
                old_dataz2.append(torch.from_numpy(z2tmp).type(torch.FloatTensor))

                with open(qp_filename2,'wb') as f:
                    pickle.dump(model2dat,f,protocol=pickle.HIGHEST_PROTOCOL)

                print('Done with model 2')
                qps_l2.append(qp2_lat)
        elif qtype == 'qp':

            vqvae_template1_save = os.path.join(root, 'vqvae_MNIST_'+ str(n_qp) + '_template1')
            vqvae_template2_save = os.path.join(root,'vqvae_MNIST_template2')
            vqvae_template1 = VQ_VAE(save_dir = vae_template1_save, z_dim = z_dim,beta=1.25,n_embeddings=n_qp)
            vqvae_template2 = VQ_VAE(save_dir = vae_template2_save, z_dim = z_dim,beta=1.25)
            template1_fname = os.path.join(vae_template1_save, 'checkpoint_200.tar')
            template2_fname = os.path.join(vae_template2_save, 'checkpoint_200.tar')

            qp_filename1 = os.path.join(qp_folder,'vqps_model1_' + str(n_qp) + '_qps.pickle')
            qp_filename2 = os.path.join(qp_folder,'vqps_model2_' + str(n_qp) + '_qps.pickle')
            qp1_tmp = []
            qp2_tmp = []

            if not os.path.isfile(template1_fname):
                vqvae_template1.train_loop(across1,epochs=201,test_freq=None,vis_freq=None)

            else:
                vqvae_template1.load_state(template1_fname)
            if not os.path.isfile(template2_fname):
                vqvae_template2.train_loop(across1,epochs=201,test_freq=None,vis_freq=None)
            else:
                vqvae_template2.load_state(template2_fname)


    if model_compare:
        vqvaes1 = []
        vqvaes2 = []

        vae_saves1 = []
        vae_saves2 = []
        vaes1 = []
        vaes2 = []


        bvaes1 = []
        bvaes2 = []


        qpvaes1 = []
        qpvaes2 = []

        across1_copy = copy.copy(across1)
        across2_copy = copy.copy(across2)

        #### Setting up things for plotting ####
        latent_inds1 = np.repeat(np.array([1,2,3]),n_train1)#,4
        latent_inds1 = np.hstack([np.zeros([n_train1,]),latent_inds1])
        latent_inds2 = np.repeat(np.array([1,2,3]),n_train2)#,4
        latent_inds2 = np.hstack([np.zeros([n_train2,]),latent_inds2])
        models = [1,2,3]#,4]
        mnames = ['Vanilla', 'bVAE','qpVAE']#'vqVAE',

        umap_transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
            metric='euclidean', random_state=42)
        pca_transform = PCA(n_components = 2)
        color_orig = 'dodgerblue'
        model_colors = ['lime','brown','darkblue','crimson']
        for qp_ind, n_qp in enumerate(n_qps):

            beta_qpvae = 2.75*n_qp/batch_size

            print('Training on N QPs = ', n_qp)
            oldx1 = old_datax1[qp_ind]
            oldx2 = old_datax2[qp_ind]
            #print(oldx1.shape)

            oldy1 = old_datay1[qp_ind]
            oldy2 = old_datay2[qp_ind]

            oldz1 = old_dataz1[qp_ind]
            oldz2 = old_dataz2[qp_ind]

            train_dataset1_copy = copy.copy(train_dataset1)
            train_dataset2_copy = copy.copy(train_dataset2)

            train_dataset1_copy.images = np.vstack([train_dataset1_copy.images,oldx2])
            train_dataset1_copy.labels = np.hstack([train_dataset1_copy.labels,oldy2])
            train_dataset2_copy.images = np.vstack([train_dataset2_copy.images,oldx1])
            train_dataset2_copy.images = np.hstack([train_dataset2_copy.labels,oldy1])

            tmp_dl1 = DataLoader(train_dataset1_copy, batch_size = batch_size, \
                shuffle = True, num_workers=num_workers)
            tmp_dl2 = DataLoader(train_dataset2, batch_size = batch_size, \
                shuffle = True, num_workers=num_workers)

            across1_copy['train'] = tmp_dl1
            across2_copy['train'] = tmp_dl2

            vqvae_save1 = os.path.join(root,'vqvae_MNIST_' + str(n_qp) + 'qp1')
            vqvae_save2 = os.path.join(root,'vqvae_MNIST_' + str(n_qp) + 'qp2')

            vae_save1 = os.path.join(root,'vae_MNIST_' + str(n_qp) + 'qp1')
            vae_save2 = os.path.join(root,'vae_MNIST_' + str(n_qp) + 'qp2')

            bvae_save1 = os.path.join(root,'bvae_MNIST_' + str(n_qp) + 'qp1')
            bvae_save2 = os.path.join(root,'bvae_MNIST_' + str(n_qp) + 'qp2')

            qpvae_save1 = os.path.join(root,'qpvae_MNIST_' + str(n_qp) + 'qp1')
            qpvae_save2 = os.path.join(root,'qpvae_MNIST_' + str(n_qp) + 'qp2')
            '''
            vqvaes1.append(VQ_VAE(save_dir = vqvae_save1, z_dim=z_dim,\
                n_embeddings=n_qp,beta=1.0,n_train=n_train1))
            vqvaes2.append(VQ_VAE(save_dir = vqvae_save2, z_dim=z_dim,\
                n_embeddings=n_qp,beta=1.0,n_train=n_train2))
            '''
            vaes1.append(VAE(save_dir = vae_save1, z_dim=z_dim))
            vaes2.append(VAE(save_dir = vae_save2, z_dim=z_dim))

            bvaes1.append(B_VAE(save_dir = bvae_save1, z_dim=z_dim, beta=beta_bvae))
            bvaes2.append(B_VAE(save_dir = bvae_save2, z_dim=z_dim, beta=beta_bvae))

            qpvaes1.append(QP_VAE(save_dir=qpvae_save1, z_dim=z_dim,no_sample = True,tau = beta_qpvae))
            qpvaes2.append(QP_VAE(save_dir=qpvae_save2, z_dim=z_dim,no_sample = True, tau = beta_qpvae))

            vae1_filename = os.path.join(vae_save1, 'checkpoint_200.tar')
            vae2_filename = os.path.join(vae_save2, 'checkpoint_200.tar')

            vqvae1_filename = os.path.join(vqvae_save1, 'checkpoint_200.tar')
            vqvae2_filename = os.path.join(vqvae_save2, 'checkpoint_200.tar')
            bvae1_filename = os.path.join(bvae_save1, 'checkpoint_200.tar')
            bvae2_filename = os.path.join(bvae_save2, 'checkpoint_200.tar')
            qpvae1_filename = os.path.join(qpvae_save1, 'checkpoint_200.tar')
            qpvae2_filename = os.path.join(qpvae_save2, 'checkpoint_200.tar')

            #### training/loading vanilla vaes ####
            print('Training/Loading Vanilla VAEs')
            if not os.path.isfile(vae1_filename):
                vaes1[qp_ind].train_loop(across1_copy,epochs=201,test_freq=None,vis_freq=5)
            else:
                vaes1[qp_ind].load_state(vae1_filename)
            if not os.path.isfile(vae2_filename):
                vaes2[qp_ind].train_loop(across2_copy,epochs=201,test_freq=None,vis_freq=5)
            else:
                vaes2[qp_ind].load_state(vae2_filename)

            #### training/loading beta vaes
            print('Training/Loading Beta VAEs')
            if not os.path.isfile(bvae1_filename):
                bvaes1[qp_ind].train_loop(across1_copy,epochs=201,test_freq=None,vis_freq=5)
            else:
                bvaes1[qp_ind].load_state(bvae1_filename)
            if not os.path.isfile(bvae2_filename):
                bvaes2[qp_ind].train_loop(across2_copy,epochs=201,test_freq=None,vis_freq=5)
            else:
                bvaes2[qp_ind].load_state(bvae2_filename)

            #### training/loading vqvaes
            print('Training/Loading VQVAEs')
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
            if not os.path.isfile(qpvae1_filename):
                qpvaes1[qp_ind].train_loop(across1,oldx2, oldz2,epochs=201,test_freq=None,vis_freq=5)
            else:
                qpvaes1[qp_ind].load_state(qpvae1_filename)
            if not os.path.isfile(qpvae2_filename):
                qpvaes2[qp_ind].train_loop(across2,oldx1, oldz1, epochs=201,test_freq=None,vis_freq=5)
            else:
                qpvaes2[qp_ind].load_state(qpvae2_filename)


            ######### Once models have been trained, we can compare their latent spaces
            ######### What are the right comparisons here?

            ######### what if we train a beta vae then compress it
            ######### can this learn the same disentangled latent space?


            ########regardless, compare latents1 to model trained on latents2, given latents1
            #use latents_dataloader1, latents_dataloader2

            #latents = model.get_latent(latents_dataloader)
            print('Getting latents from all models')
            vanilla_m1l2 = vaes1[qp_ind].get_latent(latents_dataloader2)
            vanilla_m2l1 = vaes2[qp_ind].get_latent(latents_dataloader1)

            bvae_m1l2 = bvaes1[qp_ind].get_latent(latents_dataloader2)
            bvae_m2l1 = bvaes2[qp_ind].get_latent(latents_dataloader1)
            '''
            vqvae_m1l2 = vqvaes1[qp_ind].get_latent(latents_dataloader2)
            vqvae_m2l1 = vqvaes2[qp_ind].get_latent(latents_dataloader1)
            '''
            qpvaes_m1l2 = qpvaes1[qp_ind].get_latent(latents_dataloader2)
            qpvaes_m2l1 = qpvaes2[qp_ind].get_latent(latents_dataloader1)
            print('Done!')
            all_latents1 = np.vstack([latents1,vanilla_m2l1,bvae_m2l1,qpvaes_m2l1])#vqvae_m2l1,
            all_latents2 = np.vstack([latents2,vanilla_m1l2,bvae_m1l2,qpvaes_m1l2])#vqvae_m1l2,
            print(all_latents1.shape)
            print(all_latents2.shape)
            print('UMAP transforms')
            print('latents 1')
            umap_l1 = umap_transform.fit_transform(all_latents1)
            print('latents 2')
            umap_l2 = umap_transform.fit_transform(all_latents2)

            print('PCA transforms')
            print('latents 1')
            pca_l1 = pca_transform.fit_transform(all_latents1)
            print('latents 2')
            pca_l2 = pca_transform.fit_transform(all_latents2)
            print('Done!')

            print('Making UMAP and PCA plots for all models')
            for model in models:

                mname = mnames[model-1]


                orig_inds1 = latent_inds1 == 0
                orig_inds2 = latent_inds2 == 0
                inds1 = latent_inds1 == model
                inds2 = latent_inds2 == model
                if model == 1:
                    kl_data1 = calc_dkl(vae_template1,vaes1[qp_ind],latents_dataloader1)
                    kl_data2 = calc_dkl(vae_template2,vaes2[qp_ind],latents_dataloader2)
                elif model == 2:
                    kl_data1 = calc_dkl(vae_template1,bvaes1[qp_ind],latents_dataloader1)
                    kl_data2 = calc_dkl(vae_template2,bvaes2[qp_ind],latents_dataloader2)
                elif model == 3:
                    kl_data1 = calc_dkl(vae_template1,qpvaes1[qp_ind],latents_dataloader1)
                    kl_data2 = calc_dkl(vae_template2,qpvaes2[qp_ind],latents_dataloader2)

                print('KL Divergence between template and ', mname, ' latents 1 : ', kl_data1)
                print('KL Divergence between template and ', mname, ' latents 2 : ', kl_data2)
                ##### UMAP Plots ####
                save_filename1 = os.path.join(root, mname + '_' + \
                                        str(n_qp) + 'qps_latents1_comparison_umap.pdf')
                save_filename2 = os.path.join(root, mname + '_' + \
                                         str(n_qp) + 'qps_latents2_comparison_umap.pdf')
                ax = plt.gca()

                orig = ax.scatter(umap_l1[orig_inds1,0],umap_l1[orig_inds1,1],c=color_orig,marker = 'o', alpha=0.1)
                new = ax.scatter(umap_l1[inds1,0],umap_l1[inds1,1],c=model_colors[model-1],marker='o', alpha=0.1)

                plt.legend([orig,new],['original latents 1', mname + ' latents 1'])
                plt.xlabel('UMAP Component 1')
                plt.ylabel('UMAP Component 2')
                plt.tight_layout()
                plt.axis('square')
                plt.savefig(save_filename1)
                plt.close('all')

                ax = plt.gca()

                orig = ax.scatter(umap_l2[orig_inds2,0],umap_l2[orig_inds2,1],c=color_orig,marker = 'o',alpha=0.1)
                new = ax.scatter(umap_l2[inds2,0],umap_l2[inds2,1],c=model_colors[model-1],marker='o',alpha=0.1)

                plt.legend([orig,new],['original latents 2', mname + ' latents 2'])
                plt.xlabel('UMAP Component 1')
                plt.ylabel('UMAP Component 2')
                plt.tight_layout()
                plt.axis('square')
                plt.savefig(save_filename2)
                plt.close('all')

                #### PCA Plots
                save_filename1 = os.path.join(root, mname + '_' + \
                                         str(n_qp) + 'qps_latents1_comparison_pca.pdf')
                save_filename2 = os.path.join(root, mname + '_' + \
                                         str(n_qp) + 'qps_latents2_comparison_pca.pdf')
                ax = plt.gca()

                orig = ax.scatter(pca_l1[orig_inds1,0],pca_l1[orig_inds1,1],c=color_orig,marker = 'o',alpha=0.1)
                new = ax.scatter(pca_l1[inds1,0],pca_l1[inds1,1],c=model_colors[model-1],marker='o',alpha=0.1)

                plt.legend([orig,new],['original latents 1', mname + ' latents 1'])
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.tight_layout()
                plt.axis('square')
                plt.savefig(save_filename1)
                plt.close('all')

                ax = plt.gca()

                orig = ax.scatter(pca_l2[orig_inds2,0],pca_l2[orig_inds2,1],c=color_orig,marker = 'o',alpha=0.1)
                new = ax.scatter(pca_l2[inds2,0],pca_l2[inds2,1],c=model_colors[model-1],marker='o',alpha=0.1)

                plt.legend([orig,new],['original latents 2', mname + ' latents 2'])
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.tight_layout()
                plt.axis('square')
                plt.savefig(save_filename2)
                plt.close('all')

            ####PCA all plots
            save_filename1 = os.path.join(root, \
                                     str(n_qp) + 'qps_latents1_comparison_pca_all.pdf')
            save_filename2 = os.path.join(root, \
                                     str(n_qp) + 'qps_latents2_comparison_pca_all.pdf')

            ax = plt.gca()

            orig = ax.scatter(pca_l2[orig_inds2,0],pca_l2[orig_inds2,1],c=color_orig,marker = 'o',alpha=0.1)
            van = ax.scatter(pca_l2[latent_inds2 == 1,0],pca_l2[latent_inds2 == 1,1],c=model_colors[0],marker='o',alpha=0.1)
            beta = ax.scatter(pca_l2[latent_inds2 == 2,0],pca_l2[latent_inds2 == 2,1],c=model_colors[1],marker='o',alpha=0.1)
            #vq = ax.scatter(pca_l2[latent_inds2 == 3,0],pca_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
            qp = ax.scatter(pca_l2[latent_inds2 == 3,0],pca_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o',alpha=0.1)

            plt.legend([orig,van,beta,qp],['original latents 2', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.tight_layout()
            plt.axis('square')
            plt.savefig(save_filename2)
            plt.close('all')
            print('Done!')

            ax = plt.gca()

            orig = ax.scatter(pca_l1[orig_inds1,0],pca_l1[orig_inds1,1],c=color_orig,marker = 'o',alpha=0.1)
            van = ax.scatter(pca_l1[latent_inds1 == 1,0],pca_l1[latent_inds1 == 1,1],c=model_colors[0],marker='o',alpha=0.1)
            beta = ax.scatter(pca_l1[latent_inds1 == 2,0],pca_l1[latent_inds1 == 2,1],c=model_colors[1],marker='o',alpha=0.1)
            #vq = ax.scatter(pca_l1[latent_inds1 == 3,0],pca_l1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
            qp = ax.scatter(pca_l1[latent_inds1 == 3,0],pca_l1[latent_inds1 == 3,1],c=model_colors[2],marker='o',alpha=0.1)

            plt.legend([orig,van,beta,qp],['original latents 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.tight_layout()
            plt.axis('square')
            plt.savefig(save_filename1)
            plt.close('all')
            print('Done!')

            #### UMAP all plots
            save_filename1 = os.path.join(root,  \
                                     str(n_qp) + 'qps_latents1_comparison_umap_all.pdf')
            save_filename2 = os.path.join(root, \
                                     str(n_qp) + 'qps_latents2_comparison_umap_all.pdf')

            ax = plt.gca()

            orig = ax.scatter(umap_l2[orig_inds2,0],umap_l2[orig_inds2,1],c=color_orig,marker = 'o',alpha=0.1)
            van = ax.scatter(umap_l2[latent_inds2 == 1,0],umap_l2[latent_inds2 == 1,1],c=model_colors[0],marker='o',alpha=0.1)
            beta = ax.scatter(umap_l2[latent_inds2 == 2,0],umap_l2[latent_inds2 == 2,1],c=model_colors[1],marker='o',alpha=0.1)
            #vq = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o')
            qp = ax.scatter(umap_l2[latent_inds2 == 3,0],umap_l2[latent_inds2 == 3,1],c=model_colors[2],marker='o',alpha=0.1)

            plt.legend([orig,van,beta,qp],['original latents 2', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.tight_layout()
            plt.axis('square')
            plt.savefig(save_filename2)
            plt.close('all')
            print('Done!')

            ax = plt.gca()

            orig = ax.scatter(umap_l1[orig_inds1,0],umap_l1[orig_inds1,1],c=color_orig,marker = 'o',alpha=0.1)
            van = ax.scatter(umap_l1[latent_inds1 == 1,0],umap_l1[latent_inds1 == 1,1],c=model_colors[0],marker='o',alpha=0.1)
            beta = ax.scatter(umap_l1[latent_inds1 == 2,0],umap_l1[latent_inds1 == 2,1],c=model_colors[1],marker='o',alpha=0.1)
            #vq = ax.scatter(umap_l1[latent_inds1 == 3,0],umap_l1[latent_inds1 == 3,1],c=model_colors[2],marker='o')
            qp = ax.scatter(umap_l1[latent_inds1 == 3,0],umap_l1[latent_inds1 == 3,1],c=model_colors[2],marker='o',alpha=0.1)

            plt.legend([orig,van,beta,qp],['original latents 1', mnames[0],mnames[1],mnames[2]])#,mnames[3]])
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.tight_layout()
            plt.axis('square')
            plt.savefig(save_filename1)
            plt.close('all')
            print('Done!')

    if betas_qps_compare:
        betas = list(np.arange(1,3.5,0.6))

        b_qp_comb = itertools.product(betas, n_qps)

        latent_inds1 = [np.zeros([n_train1,])]
        latent_inds2 = [np.zeros([n_train2,])]
        models = [1,2,3]#,4]
        mnames = ['Vanilla', 'bVAE','qpVAE']#'vqVAE',

        umap_transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
            metric='euclidean', random_state=42)
        pca_transform = PCA(n_components = 2)
        color_orig = 'dodgerblue'
        #model_colors = ['lime','brown','darkblue','crimson']

        latents1_all = [latents1]
        latents2_all = [latents2]
        model_ind = 1
        qpvaes1 = []
        qpvaes2 = []

        for b_qp in b_qp_comb:

            print(model_ind)


            (beta, n_qp) = b_qp

            qp_ind = n_qps.index(n_qp)
            #print(f'M = {tmpM:d}, d2 = {tmpd2:d}')
            print(f'Training on N QPs = {n_qp:d}, beta = {beta:.3f}')
            oldx1 = old_datax1[qp_ind]
            oldx2 = old_datax2[qp_ind]
            #print(oldx1.shape)

            oldy1 = old_datay1[qp_ind]
            oldy2 = old_datay2[qp_ind]

            oldz1 = old_dataz1[qp_ind]
            oldz2 = old_dataz2[qp_ind]

            qpvae_save1 = os.path.join(root,'qpvae_MNIST_' + str(n_qp) + 'qp1_beta_' + str(beta))
            qpvae_save2 = os.path.join(root,'qpvae_MNIST_' + str(n_qp) + 'qp2_beta_' + str(beta))
            '''
            vqvaes1.append(VQ_VAE(save_dir = vqvae_save1, z_dim=z_dim,\
                n_embeddings=n_qp,beta=1.0,n_train=n_train1))
            vqvaes2.append(VQ_VAE(save_dir = vqvae_save2, z_dim=z_dim,\
                n_embeddings=n_qp,beta=1.0,n_train=n_train2))
            '''

            qpvaes1.append(QP_VAE(save_dir=qpvae_save1, z_dim=z_dim,zhat = True,tau = beta))
            qpvaes2.append(QP_VAE(save_dir=qpvae_save2, z_dim=z_dim,zhat = True, tau = beta))
            qpvae1_filename = os.path.join(qpvae_save1, 'checkpoint_200.tar')
            qpvae2_filename = os.path.join(qpvae_save2, 'checkpoint_200.tar')
            print('Training/Loading QP-VAEs')
            if not os.path.isfile(qpvae1_filename):
                qpvaes1[model_ind - 1].train_loop(across1,oldx2, oldz2,epochs=201,test_freq=None,vis_freq=5)
            else:
                qpvaes1[model_ind - 1].load_state(qpvae1_filename)
            if not os.path.isfile(qpvae2_filename):
                qpvaes2[model_ind - 1].train_loop(across2,oldx1, oldz1, epochs=201,test_freq=None,vis_freq=5)
            else:
                qpvaes2[model_ind - 1].load_state(qpvae2_filename)

            tmp1L = qpvaes2[qp_ind].get_latent(latents_dataloader1)
            tmp2L = qpvaes1[qp_ind].get_latent(latents_dataloader2)
            latents1_all.append(tmp1L)
            latents2_all.append(tmp2L)

            latent_inds1.append(model_ind * np.ones([tmp1L.shape[0],]))
            latent_inds2.append(model_ind * np.ones([tmp2L.shape[0],]))

            print(tmp1L.shape)
            print(tmp2L.shape)

            model_ind += 1
        latents1_mat = np.vstack(latents1_all)
        latents2_mat = np.vstack(latents2_all)

        #print(list(b_qp_comb))
        #print(list(range(model_ind)))

        print(latents1_mat.shape)
        print(latents2_mat.shape)
        latent_inds_vec1 = np.hstack(latent_inds1)
        latent_inds_vec2 = np.hstack(latent_inds2)
        print(latent_inds_vec1.shape)
        print(latent_inds_vec2.shape)

        print('UMAP transforms')
        print('latents 1')
        umap_l1 = umap_transform.fit_transform(latents1_mat)
        print('latents 2')
        umap_l2 = umap_transform.fit_transform(latents2_mat)

        print('PCA transforms')
        print('latents 1')
        pca_l1 = pca_transform.fit_transform(latents1_mat)
        print('latents 2')
        pca_l2 = pca_transform.fit_transform(latents2_mat)
        print('Done!')

        print(umap_l1.shape)
        print(umap_l2.shape)
        f1 = plt.figure(1)
        f2 = plt.figure(2)
        f3 = plt.figure(3)
        f4 = plt.figure(4)
        ax1 = f1.gca()
        ax2 = f2.gca()
        ax3 = f3.gca()
        ax4 = f4.gca()

        save_filename1 = os.path.join(root, 'qps_latents1_betas_umap.pdf')
        save_filename2 = os.path.join(root, 'qps_latents2_betas_umap.pdf')
        save_filename3 = os.path.join(root, 'qps_latents1_betas_pca.pdf')
        save_filename4 = os.path.join(root, 'qps_latents2_betas_pca.pdf')
        b_qp_comb = list(itertools.product(betas, n_qps))
        for ind in range(model_ind - 1):
            (beta, n_qp) = b_qp_comb[ind]

            inds1 = latent_inds_vec1 == ind
            inds2 = latent_inds_vec2 == ind

            ##### UMAP Plots ####

            ax1.scatter(umap_l1[inds1,0],umap_l1[inds1,1],\
                    marker='o', alpha=0.1, \
                    label = 'N QP: ' + str(n_qp) + ', beta = ' + str(beta))

            ax2.scatter(umap_l2[inds2,0],umap_l2[inds2,1],\
                    marker='o', alpha=0.1, \
                    label = 'N QP: ' + str(n_qp) + ', beta = ' + str(beta))

            ax3.scatter(pca_l1[inds1,0],pca_l1[inds1,1],\
                    marker='o', alpha=0.1, \
                    label = 'N QP: ' + str(n_qp) + ', beta = ' + str(beta))

            ax4.scatter(pca_l2[inds2,0],pca_l2[inds2,1],\
                    marker='o', alpha=0.1, \
                    label = 'N QP: ' + str(n_qp) + ', beta = ' + str(beta))


        plt.legend(fontsize='small')
        ax1.set_xlabel('UMAP Component 1')
        ax2.set_xlabel('UMAP Component 1')
        ax3.set_xlabel('PC 1')
        ax4.set_xlabel('PC 1')

        ax1.set_ylabel('UMAP Component 2')
        ax2.set_ylabel('UMAP Component 2')
        ax3.set_ylabel('PC 2')
        ax4.set_ylabel('PC 2')

        #f1.tight_layout()
        #f1.axis('square')
        #f2.tight_layout()
        #f2.axis('square')
        #f3.tight_layout()
        #f3.axis('square')
        #f4.tight_layout()
        #f4.axis('square')

        f1.savefig(save_filename1)
        f2.savefig(save_filename2)
        f3.savefig(save_filename3)
        f4.savefig(save_filename4)

        plt.close('all')
