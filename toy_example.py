#### TOY MODEL CODE ######


#goal train VAEs on subsets of data. we want to compare between vanilla VAE, VQ-VAE, and our method

#specifically, this is with the mnist dataset
# we will train on a subset of the digits, then on a random subset of the data across digits

#umap latent space for each model, compare all of them


###### Imports ################
from ava.models.vae_MNIST import VAE
from ava.models.dw_MNIST import dw_VAE
from ava.models.qp_vae_MNIST import QP_VAE
from ava.models.vq_vae_MNIST import VQ_VAE
from ava.models.b_vae_MNIST import B_VAE
from ava.plotting import grid_plot
import matplotlib.pyplot as plt

import numpy as np
import os
import copy
import h5py
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import DecisionTreeRegressor
import torch
import shutil
import gzip
import umap

import tensorflow as tf
from torch.utils.data import Dataset, DataLoader



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


##################################
# Import MNIST
##################################
if __name__ == '__main__':
    root = '/home/mmartinez/autoencoded-vocal-analysis/ava_qpe_experiments'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist = tf.keras.datasets.mnist
    #loads these in as np arrays- much better than torch smh
    (x_train,y_train), (x_test,y_test) = mnist.load_data()

    x_all = np.vstack((x_train,x_test)).astype(np.float32)/255
    y_all = np.hstack((y_train,y_test)).astype(np.int)
    #print(x_all.shape)

    unique_labels = np.unique(y_all)
    #print('im here')

##################################
# Split MNIST into random subsets (across digits)
# randomly select 1/4 of data, then a separate quarter
# The end product of this should be a torch dataloader
# with the required train, test splits. maybe use a custom class
# building on torch DataLoader?
##################################
# 0.25 train/test split

    batch_size = 100
    num_workers = min(7,os.cpu_count()-1)

    split = 0.65
    n_dat = len(y_all)

    n_train = int(round(split * n_dat))
    n_test = n_dat - n_train
    np.random.seed(35)
    order1 = np.random.permutation(n_dat)
    order2 = np.random.permutation(n_dat)

    x1 = x_all[order1,:,:]
    x2 = x_all[order2,:,:]

    y1 = y_all[order1,]
    y2 = y_all[order2,]
    #print(y1[0:15])

    x_train1 = x1[0:n_train,:,:]
    y_train1 = y1[0:n_train,]

    x_test1 = x1[n_train::,:,:]
    y_test1 = y1[n_train::,]



    x_train2 = x2[0:n_train,:,:]
    y_train2 = y2[0:n_train,]

    x_test2 = x2[n_train::,:,:]
    y_test2 = y2[n_train::,]

    train_dataset_across = MNISTDataset((x_train1,y_train1))
    test_dataset1 = MNISTDataset((x_test1,y_test1))
    #print(x_train1.shape)
    #print(x_train1.type)
    train_across_dataloader1 = DataLoader(train_dataset_across, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)
    latents_dataloader1 = DataLoader(train_dataset_across, batch_size = batch_size, \
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
    latents_dataloader2 = DataLoader(train_dataset_across, batch_size = batch_size, \
        shuffle = False, num_workers=num_workers)

    across2 = {'train': train_across_dataloader2, 'test': test_across_dataloader2}

    data_across_dig = {
        'train_dat1': (x_train1,y_train1),
        'train_dat2': (x_train2,y_train2),
        'test_dat1': (x_test1,y_test1),
        'test_dat2': (x_test2,y_test2)
        }

##################################
# Split MNIST into random subsets (on digits)
# subset 1, subset 2
# maybe randomly select 5/10 digits, pick other half, then use half to train, half to test
##################################
# 0.25 train/test split?

# train digits 1 become test digits 2
#jk no they dont
    '''
    dig_split = 0.65 # 5/10 digits
    train_split = 0.65

    np.random.seed(42)
    new_dig = np.random.permutation(unique_labels)

#new_dig = unique_labels[dig_ord]
    dig_train1 = new_dig[0:5]
    dig_train2 = new_dig[-5::]

    train_dig1 = np.vstack([x_all[y_all == label,:,:] for label in dig_train1])
    train_dig2 = np.vstack([x_all[y_all == label,:,:] for label in dig_train2])

    train_lab1 = np.hstack([y_all[y_all == label,] for label in dig_train1])
    train_lab2 = np.hstack([y_all[y_all == label,] for label in dig_train2])

    n_train1 = int(round(train_split*len(train_lab1)))
    n_train2 = int(round(train_split*len(train_lab2)))

    order1 = np.random.permutation(len(train_lab1))
    order2 = np.random.permutation(len(train_lab2))

    train_dig1 = train_dig1[order1,:,:]
    train_lab1 = train_lab1[order1,]

    train_dig2 = train_dig2[order2,:,:]
    train_lab2 = train_lab2[order2,]

    x_train1 = train_dig1[0:n_train1,:,:]
    y_train1 = train_lab1[0:n_train1,]
    x_test1 = train_dig1[n_train1::,:,:]
    y_test1 = train_lab1[n_train::,]

    x_train2 = train_dig2[0:n_train2,:,:]
    y_train2 = train_lab2[0:n_train2,]
    x_test2 = train_dig2[n_train2::,:,:]
    y_test2 = train_lab2[n_train::,]

    train_dataset1 = MNISTDataset((x_train1,y_train1),transform = torch.tensor)
    test_dataset1 = MNISTDataset((x_test1,y_test1), transform=torch.tensor)

    train_win_dataloader1 = DataLoader(train_dataset1, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)
    test_win_dataloader1 = DataLoader(test_dataset1, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)

    within1 = {'train':train_win_dataloader1, 'test':test_win_dataloader1}

    train_dataset2 = MNISTDataset((x_train2,y_train2))
    test_dataset2 = MNISTDataset((x_test2,y_test2))

    train_win_dataloader2 = DataLoader(train_dataset2, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)
    test_win_dataloader2 = DataLoader(test_dataset2, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)

    within2 = {'train':train_win_dataloader2, 'test':test_win_dataloader2}

    data_win_dig = {
        'train_dat1': (x_train1, y_train1),
        'train_dat2': (x_train2, y_train2),
        'test_dat1': (x_test1, y_test1),
        'test_dat2': (x_test2, y_test2)
        }
'''
######################
# Great! we have split our data into two sets: train and test
# now we want to initialize our models.
# We want two VQ-VAEs, two vanilla VAEs, two of our version
# let's just do 2 at a time to minimize worklaod
######################
    n_qps = 128
    z_dim_vq = 49 # 7x7 space
    z_dim_vae = 64
    beta = 0.5

    vqvae1_save = os.path.join(root, 'vqvae_MNIST_MA_128qp')
    vqvae2_save = os.path.join(root, 'vqvae_MNIST_2stage_rand')

    bvae1_save = os.path.join(root,'bvae_MNIST_test_b20')

    vae1_save = os.path.join(root, 'vae_for_qp')
    vae2_save = os.path.join(root, 'vae_across_dig2')

    qpvae_save1 = os.path.join(root, 'qpvae_test_128qp')
    govae2_save = os.path.join(root, 'govae_across_dig2')

    vqvae1 = VQ_VAE(save_dir = vqvae1_save, z_dim = z_dim_vq, n_embeddings = n_qps,beta=1.0,n_train=n_train)
    vqvae2 = VQ_VAE(save_dir = vqvae2_save, z_dim = z_dim_vq, n_embeddings = n_qps,beta=1.0)

    vae1 = VAE(save_dir = vae1_save, z_dim = z_dim_vae)
    vae2 = VAE(save_dir = vae2_save, z_dim = z_dim_vae)

    qpvae1 = QP_VAE(save_dir = qpvae_save1, use_old = 1, use_new = 1, \
        beta = 2.75, zhat = True, z_dim = z_dim_vae)
    #govae2 = goVAE(save_dir = govae2_save, use_old = 1, use_new = 1, \
    #    beta = beta, zhat = True, z_dim = z_dim_vae)

    bvae1 = B_VAE(save_dir = bvae1_save, z_dim = z_dim_vq,beta=20)

    model_filename = os.path.join(vae1_save, 'checkpoint_200.tar')
    if not os.path.isfile(model_filename):

        vae1.train_loop(across1,epochs=201,test_freq=None,vis_freq=5)
        #vqvae2.train_loop(across1,epochs=201,test_freq=None,vis_freq=5)

    else:
        vae1.load_state(model_filename)
        #vqvae1.train_loop(across1,epochs=71,test_freq=None,vis_freq=5)
        #vqvae2.load_state(os.path.join(vqvae2_save, 'checkpoint_400.tar'))
        #vqvae1.train_loop(across1,epochs=800,test_freq=None,vis_freq=1)

    latents = vae1.get_latent(latents_dataloader1)
    qp_lat = KMeans(n_clusters=n_qps).fit(latents)
    qps = qp_lat.cluster_centers_
    qp_closest = []

    for centroid_n, qp in enumerate(qps):
        cent_dists = np.linalg.norm(qp - latents, axis=1)
        min_dist = np.argmin(cent_dists)
        qp_closest.append(min_dist)

    old_datax = torch.from_numpy(x_train1[qp_closest,:,:]).type(torch.FloatTensor)
    old_datay = y_train1[qp_closest,]
    print('Numbers in centroids: ', np.unique(old_datay))

    oldz = torch.from_numpy(latents[qp_closest,:]).type(torch.FloatTensor)
    model_filename_qp = os.path.join(qpvae_save1,'checkpoint_200.tar')

    if not os.path.isfile(model_filename_qp):

        qpvae1.train_loop(across1,old_datax,oldz,epochs=201,test_freq=None,vis_freq=5)
        #vqvae2.train_loop(across1,epochs=201,test_freq=None,vis_freq=5)

    else:
        qpvae1.load_state(model_filename)
    #qps = vqvae1.get_qps()

    #latents = vqvae1.get_latent(latents_dataloader1)

    #c1 = AgglomerativeClustering(n_clusters = 128) # 7 bits of info
    #c1.fit(latents)
    #transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
    #    metric='euclidean', random_state=42)

    #embed = transform.fit_transform(np.vstack([qps,latents]))

    '''
    mult = int(qps.shape[0]/8)
    for ii in range(8):
        ax = plt.gca()
        #print(embed[mult*ii:mult*(ii+1),0].shape)

        latents = ax.scatter(embed[:,0], embed[:,1],c='b', marker='.')
        qpsTmp = ax.scatter(embed[mult*ii:mult*(ii+1),0],embed[mult*ii:mult*(ii+1),1],c='r',marker='o')
        save_filename=os.path.join(vqvae1_save, \
                                    'umap_vqvae_l_qp_' + str(ii*mult) + 'to_' + str((ii+1)*mult - 1) + '.pdf')
        plt.legend([qpsTmp,latents],['qps ' + str(ii*mult) + ' to ' + str((ii+1)*mult - 1), 'latents'])
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.tight_layout()
        plt.axis('square')
        plt.savefig(save_filename)
        plt.close('all')

    #qp_recs = vqvae1.decode(torch.from_numpy(qps).to(vqvae1.device)).detach().cpu().numpy()
#    vqvae1.visualize(across1['train'])
    vqvae1.vis_qp_recs()


#    model_filename = os.path.join(vae1_save, 'checkpoint_100.tar')
#    if not os.path.isfile(model_filename):
#        vae1.train_loop(across1,epochs=201,test_freq=None,vis_freq=1)
#    else:
#        vae1.load_state(model_filename)
    '''
    '''
    Dai & Wipf VAE
    '''
    '''
    stage1_dir = os.path.join(root, 'vae_stage1')
    stage2_dir = os.path.join(root, 'vae_stage2')

    # Training stage 1
    stage2_dl =  DataLoader(train_dataset_across, batch_size = batch_size, \
        shuffle = False, num_workers=num_workers)
    lr = 0.0001
    vae_stage1 = dw_VAE(save_dir = stage1_dir, z_dim = z_dim_vae, stage = 1, lr=lr)
    model_filename_s1 = os.path.join(stage1_dir, 'checkpoint_200.tar')

    if not os.path.isfile(model_filename_s1):
        print('Training Stage 1')
        vae_stage1.train_loop(across1, epochs=151, test_freq=None, vis_freq=1)

        vae_stage1.lr = lr/2
        vae_stage1.optimizer.lr = lr/2
        vae_stage1.train_loop(across1,epochs=150, test_freq=None, vis_freq=1)

        vae_stage1.lr = lr/4
        vae_stage1.optimizer.lr = lr/4
        vae_stage1.train_loop(across1,epochs=100, test_freq=None, vis_freq=1)


    else:

        vae_stage1.load_state(model_filename_s1)

    print('Getting Latents from Stage 1')

    latents_train_s2 = np.expand_dims(vae_stage1.get_latent(stage2_dl),axis=1).astype('float32')
    latent_labels_train_s2 = y_train1
    latents_test_s2 = np.expand_dims(vae_stage1.get_latent(test_across_dataloader1), axis=1).astype('float32')
    latent_labels_test_s2 = y_test1

    train_s2 = MNISTDataset((latents_train_s2,latent_labels_train_s2))
    test_s2 = MNISTDataset((latents_test_s2,latent_labels_test_s2))

    train_s2_dl = DataLoader(train_s2, batch_size = batch_size, \
        shuffle = True, num_workers=num_workers)
    test_s2_dl = DataLoader(test_s2, batch_size = batch_size, \
         shuffle = False, num_workers = num_workers)

    across_s2 = {'train':train_s2_dl, 'test': test_s2_dl}

    vae_stage2 = dw_VAE(save_dir = stage2_dir, z_dim = z_dim_vae, stage = 2, lr = lr)

    model_filename_s2 = os.path.join(stage2_dir, 'checkpoint_200.tar')

    if not os.path.isfile(model_filename_s2):
        print('Training Stage 2')
        vae_stage2.train_loop(across_s2, epochs=301, test_freq=None, vis_freq=1)

        vae_stage2.lr = lr/2
        vae_stage2.optimizer.lr = lr/2
        vae_stage2.train_loop(across1,epochs=300, test_freq=None, vis_freq=1)

        vae_stage2.lr = lr/4
        vae_stage2.optimizer.lr = lr/4
        vae_stage2.train_loop(across1,epochs=200, test_freq=None, vis_freq=1)
    else:

        vae_stage2.load_state(model_filename_s2)
    print(vae_stage1.model_precision)
    print(vae_stage2.model_precision)
    '''
    '''
    #qps = vqvae1.get_qps()
    vis_dataloader_1 = DataLoader(train_dataset_across, batch_size = batch_size, \
        shuffle = False, num_workers=num_workers)
    #vqvae1.visualize(vis_dataloader_1, save_filename='reconstruction_no_qps.pdf')

    #embed = vqvae1.get_latent(vis_dataloader_1)
    #vae_latents = vae1.get_latent(vis_dataloader_1)
    transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
        metric='euclidean', random_state=42)

    #embed3 = transform.fit_transform(np.vstack([qps,embed]))
    embed = transform.fit_transform(embed)
    #embed2 = transform.fit_transform(qps)
    ax = plt.gca()
#for ind1 in range(n_encoded):
    #ind2 = ind1 + n_encoded
    #ax.plot([embed[(ind_vec == 1),0],embed[(ind_vec == 2),0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=1)
    lines = []
    labels = []
    x_test, y_test = data_across_dig['train_dat1']

    print(np.shape(embed))
    print(np.shape(x_test))
    for label in unique_labels:
        print(label)
        print(sum(y_test == label))
        print(y_test[0:15])
        a1 = ax.scatter(embed[y_test == label,0],embed[y_test == label,1],marker='o')
        lines.append(a1)
        labels.append(str(label))

    save_filename=os.path.join(vqvae1_save,'umap_vqvae_latents512_b3.pdf')
    print("total number of key points plotted: "+ str(np.shape(embed)[0]))
    plt.legend(lines,labels)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.tight_layout()
    plt.axis('square')
    plt.savefig(save_filename)
    plt.close('all')

    #ax = plt.gca()
    #for ind1 in range(n_encoded):
    #ind2 = ind1 + n_encoded
    #ax.plot([embed[(ind_vec == 1),0],embed[(ind_vec == 2),0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=1)
    '''
    '''
	lines = []
    labels = []

    a1 = ax.scatter(embed2[:,0],embed2[:,1],marker='o')



    save_filename=os.path.join(vqvae1_save,'umap_vqvae_qps512_b3.pdf')
    print("total number of key points plotted: "+ str(np.shape(embed2)[0]))
    plt.legend([a1],['qps'])
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.tight_layout()
    plt.axis('square')
    plt.savefig(save_filename)
    plt.close('all')

    ax = plt.gca()
    #for ind1 in range(n_encoded):
    #ind2 = ind1 + n_encoded
    #ax.plot([embed[(ind_vec == 1),0],embed[(ind_vec == 2),0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=1)
    lines = []
    labels = []

    qps = ax.scatter(embed2[:,0],embed2[:,1],color='k')
    latents = ax.scatter(embed3[n_qps::,0],embed3[n_qps::,1],color='b')



    save_filename=os.path.join(vqvae1_save,'umap_vqvae_qps_latents512_b3.pdf')
    print("total number of key points plotted: "+ str(np.shape(embed3)[0]))
    plt.legend([qps,latents],['qps','latents'])
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.tight_layout()
    plt.axis('square')
    plt.savefig(save_filename)
    plt.close('all')
    '''
