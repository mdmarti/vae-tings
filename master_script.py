################# Master Script for Rosetta VAE Experiments #####################


####### Imports ######
import pickle
import os
from toy_model_script import toy_model_script
from mnist_model_script import mnist_model_script
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
'''
This is where my organized thoughts for rosettaVAE experiments will go.

here is what we need:

Dataset:
    -Toy
        Joint Training
            1. Train Template
            2. Extract n qps
            3. N models
                a. for each model, output MSE template latents, linearly transformed new latents
                b. for each model, output for each latent point: logdet of covariance matrix
        Sequential Training
            1. Train Template
            2. Extract n qps
            3. N models
                a. for each model, output MSE template latents, linearly transformed new latents
                b. for each model, output for each latent point: logdet of covariance matrix
    -MNIST
        Joint Training

        Sequential Training

    -Imagenet
        Joint Training

        Sequential Training
    -Bird
        Joint Training

        Sequential Training

'''

###### Toy Model Training ########

n_train_runs = 10

'''
each of these should return
    MSE_matj: n_latent_points x n_train_runs x n_qp x n_model types for joint template
            each layer in 4th dimension is model type
            first layer: vanilla vae
            second layer: beta vae
            third layer: rosetta VAE s

    logdet_matj: n_latent_points  x n_qp x n model types for jointRosetta
            each layer in 3rd dimension is model type
            first layer: vanilla vae
            second layer: beta vae
            third layer: rosetta VAE s
    MSE_mats: n_latent_points x n_train_runs x n_qp x n_model types for sequential
            each layer in 4th dimension is model type
            first layer: vanilla vae
            second layer: beta vae
            third layer: rosetta VAE s

    logdet_mats: n_latent_points x n_qp x n model types for sequential
            each layer in 3rd dimension is model type
            first layer: vanilla vae
            second layer: beta vae
            third layer: rosetta VAE s
'''
root = '/home/mrmiews/Desktop/Pearson_Lab/master'
if not os.path.isdir(root):
    os.mkdir(root)

toypath = os.path.join(root,'toy_models')
if not os.path.isdir(toypath):
    os.mkdir(toypath)

mnistpath = os.path.join(root,'mnist_models')
if not os.path.isdir(mnistpath):
    os.mkdir(mnistpath)

toyds_name = os.path.join(root,'toy_ds.pickle')
print('Doing toy dataset experiments')
if not os.path.isfile(toyds_name):
    MSE_mat_toyj, logdet_mat_toyj, MSE_mat_toys,logdet_mat_toys = \
                    toy_model_script(n_train_runs = n_train_runs,root=toypath)
    toydat_dict = {'mse_j':MSE_mat_toyj,'mse_s':MSE_mat_toys,'logdet_j':logdet_mat_toyj,'logdet_s':logdet_mat_toys}
    with open(toyds_name,'wb') as f:
        pickle.dump(toydat_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(toyds_name,'rb') as f:
        data_toy = pickle.load(f)
    MSE_mat_toyj = data_toy['mse_j']
    MSE_mat_toys = data_toy['mse_s']
    logdet_mat_toyj = data_toy['logdet_j']
    logdet_mat_toys = data_toy['logdet_s']
print('Done!')

mnistds_name = os.path.join(root,'mnist_ds.pickle')
print('Doing MNIST dataset experiments')

if not os.path.isfile(mnistds_name):
    MSE_mat_mnistj, logdet_mat_mnistj, MSE_mat_mnists, logdet_mat_mnists = \
                    mnist_model_script(n_train_runs = n_train_runs,root = mnistpath)
    mnist_dict = {'mse_j':MSE_mat_mnistj,'mse_s':MSE_mat_mnists,'logdet_j':logdet_mat_mnistj,'logdet_s':logdet_mat_mnists}

    with open(mnistds_name,'wb') as f:
        pickle.dump(mnist_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(mnistds_name,'rb') as f:
        data_mnist = pickle.load(f)
        MSE_mat_mnistj = data_mnist['mse_j']
        MSE_mat_mnists = data_mnist['mse_s']
        logdet_mat_mnistj = data_mnist['logdet_j']
        logdet_mat_mnists = data_mnist['logdet_s']
print('Done!')

'''
MSE_mat_imagenetj, logdet_mat_imagenetj, MSE_mat_imagenets, logdet_mat_imagenets = \
                    imagenet_model_script(n_train_runs = n_train_runs)
'''
###### Same for birdsong


###### Analysis: plot distortion across points, runs, n_qps
                #### maybe bar graph: n_models bars/group, n_qp groups

toy_mse_meansj = np.mean(np.mean(MSE_mat_toyj,axis=1),axis=0)
toy_mse_meanss = np.mean(np.mean(MSE_mat_toys,axis=1),axis=0)

toy_msej_reshape = np.reshape(MSE_mat_toyj,(-1,5,3))
toy_mses_reshape = np.reshape(MSE_mat_toys,(-1,5,3))

mnist_msej_reshape = np.reshape(MSE_mat_mnistj,(-1,3,3))
mnist_mses_reshape = np.reshape(MSE_mat_mnists,(-1,3,3))

#toy_msej_reshapem = np.mean(toy_msej_reshape,axis=0)
'''
if not np.all(toy_msej_reshapem == toy_mse_meansj):
    print('this reshaping doesnt work!')
    for ii in range(5):
        for jj in range(3):
            print(toy_msej_reshapem[ii,jj])
            print(toy_mse_meansj[ii,jj])
'''


toy_logdet_meansj = np.mean(logdet_mat_toyj,axis=0)
toy_logdet_meanss = np.mean(logdet_mat_toys,axis=0)


if not toy_mse_meansj.shape == (5,3):
    print('somethings up!')
    print(MSE_mat_toyj.shape)
    print(toy_mse_meansj.shape)

n_qp_t = 5
n_qp_m = 3
n_models = 3
plt.switch_backend('Agg')

mse_figj = plt.figure(1)
logdet_figj = plt.figure(2)
n_qps_t = [2,4,8,16,6400]
n_qps_m = [32,64,20718]#[64,128]

sns.set_theme(style="whitegrid")

#tmsej_d = pd.DataFrame(toy_msej_reshape, columns = '')
for qp_ind in range(n_qp_t):
    #for m_ind in range(n_models):
    #plt.figure(figsize=(8.5,11))
    tmp_msej = np.log(toy_msej_reshape[:,qp_ind,:])
    tmp_logdetj = logdet_mat_toyj[:,qp_ind,:]/10

    tmp_msepd = pd.DataFrame(tmp_msej, columns = ['VAE','BVAE','RVAE'])
    tmp_msepd['lat_id'] = tmp_msepd.index
    tmp_msepd = pd.melt(tmp_msepd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='log_distortion')

    tmp_ldpd = pd.DataFrame(tmp_logdetj, columns = ['VAE','BVAE','RVAE'])
    tmp_ldpd['lat_id'] = tmp_ldpd.index
    tmp_ldpd = pd.melt(tmp_ldpd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='logdet')

    mse_figj, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    plt.subplots_adjust(wspace=0.6)

    #maxj = plt.gca()
    ax1 = sns.violinplot(x='Model',y='log_distortion',hue='Model',data=tmp_msepd,ax=ax1)
    ax1.set_title('distortion toy')
#    ax1.legend()
    #ax1.set_xlabel('distortion (MSE)')


    ax2 = sns.violinplot(x='Model',y='logdet',hue='Model',data=tmp_ldpd,ax=ax2)
    ax2.set_title('logdet toy, n qp = ' +  str(n_qps_t[qp_ind]))
    #ax2.legend()
    #ax2.set_xlabel('logdet')

#    maxj.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    plt.savefig('logv_results_toy_nqp' + str(n_qps_t[qp_ind]) + '.png')
    plt.close('all')
    #plt.bar(m_ind*(n_qp + 2) + np.arange(1,n_qp + 1), toy_logdet_meansj[:,m_ind])

#mse_figj = plt.figure(1)
#mse_figs = plt.figure(3)
#logdet_figs = plt.figure(4)

for qp_ind in range(n_qp_m):

    mse_figj, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    tmp_msej = np.log(mnist_msej_reshape[:,qp_ind,:])
    tmp_logdetj = logdet_mat_mnistj[:,qp_ind,:]/10

    tmp_msepd = pd.DataFrame(tmp_msej, columns = ['VAE','BVAE','RVAE'])
    tmp_msepd['lat_id'] = tmp_msepd.index
    tmp_msepd = pd.melt(tmp_msepd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='log_distortion')

    tmp_ldpd = pd.DataFrame(tmp_logdetj, columns = ['VAE','BVAE','RVAE'])
    tmp_ldpd['lat_id'] = tmp_ldpd.index
    tmp_ldpd = pd.melt(tmp_ldpd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='logdet')

    plt.subplots_adjust(wspace=0.6)
    #maxj = plt.gca()
    mse_figj, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    plt.subplots_adjust(wspace=0.6)

    #maxj = plt.gca()
    ax1 = sns.violinplot(x='Model',y='log_distortion',hue='Model',data=tmp_msepd,ax=ax1)
    ax1.set_title('distortion mnist')
#    ax1.legend()
    #ax1.set_xlabel('distortion (MSE)')


    ax2 = sns.violinplot(x='Model',y='logdet',hue='Model',data=tmp_ldpd,ax=ax2)
    ax2.set_title('logdet mnist, n qp = ' +  str(n_qps_m[qp_ind]))
    #ax2.legend()
    #ax2.set_xlabel('logdet')


#    maxj.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    plt.savefig('logv_results_MNIST_nqp' + str(n_qps_m[qp_ind]) + '.png')
    plt.close('all')
#plt.show()

for qp_ind in range(n_qp_t):
    #for m_ind in range(n_models):
    #plt.figure(figsize=(8.5,11))
    tmp_mses = np.log(toy_mses_reshape[:,qp_ind,:])
    tmp_logdets = logdet_mat_toys[:,qp_ind,:]/10

    tmp_msepd = pd.DataFrame(tmp_mses, columns = ['VAE','BVAE','RVAE'])
    tmp_msepd['lat_id'] = tmp_msepd.index
    tmp_msepd = pd.melt(tmp_msepd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='log_distortion')

    tmp_ldpd = pd.DataFrame(tmp_logdets, columns = ['VAE','BVAE','RVAE'])
    tmp_ldpd['lat_id'] = tmp_ldpd.index
    tmp_ldpd = pd.melt(tmp_ldpd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='logdet')

    mse_figj, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    plt.subplots_adjust(wspace=0.6)

    #maxj = plt.gca()
    ax1 = sns.violinplot(x='Model',y='log_distortion',hue='Model',data=tmp_msepd,ax=ax1)
    ax1.set_title('distortion toy')
#    ax1.legend()
    #ax1.set_xlabel('distortion (MSE)')


    ax2 = sns.violinplot(x='Model',y='logdet',hue='Model',data=tmp_ldpd,ax=ax2)
    ax2.set_title('logdet toy, n qp = ' +  str(n_qps_t[qp_ind]))
    #ax2.legend()
    #ax2.set_xlabel('logdet')

#    maxj.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    plt.savefig('logv_results_toys_nqp' + str(n_qps_t[qp_ind]) + '.png')
    plt.close('all')
    #plt.bar(m_ind*(n_qp + 2) + np.arange(1,n_qp + 1), toy_logdet_meansj[:,m_ind])

#mse_figj = plt.figure(1)
#mse_figs = plt.figure(3)
#logdet_figs = plt.figure(4)

for qp_ind in range(n_qp_m):

    mse_figj, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    tmp_mses = np.log(mnist_mses_reshape[:,qp_ind,:])
    tmp_logdets = logdet_mat_mnists[:,qp_ind,:]/10

    tmp_msepd = pd.DataFrame(tmp_mses, columns = ['VAE','BVAE','RVAE'])
    tmp_msepd['lat_id'] = tmp_msepd.index
    tmp_msepd = pd.melt(tmp_msepd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='log_distortion')

    tmp_ldpd = pd.DataFrame(tmp_logdets, columns = ['VAE','BVAE','RVAE'])
    tmp_ldpd['lat_id'] = tmp_ldpd.index
    tmp_ldpd = pd.melt(tmp_ldpd, id_vars=['lat_id'],value_vars=['VAE','BVAE','RVAE'],var_name='Model',value_name='logdet')

    plt.subplots_adjust(wspace=0.6)
    #maxj = plt.gca()
    mse_figj, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    plt.subplots_adjust(wspace=0.6)

    #maxj = plt.gca()
    ax1 = sns.violinplot(x='Model',y='log_distortion',hue='Model',data=tmp_msepd,ax=ax1)
    ax1.set_title('distortion mnist')
#    ax1.legend()
    #ax1.set_xlabel('distortion (MSE)')


    ax2 = sns.violinplot(x='Model',y='logdet',hue='Model',data=tmp_ldpd,ax=ax2)
    ax2.set_title('logdet mnist, n qp = ' +  str(n_qps_m[qp_ind]))
    #ax2.legend()
    #ax2.set_xlabel('logdet')


#    maxj.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    plt.savefig('logv_results_MNISTs_nqp' + str(n_qps_m[qp_ind]) + '.png')
    plt.close('all')
'''
MSE_matj: n_latent_points x n_train_runs x n_qp x n_model types for joint template
        each layer in 4th dimension is model type
        first layer: vanilla vae
        second layer: beta vae
        third layer: rosetta VAE s

logdet_matj: n_latent_points  x n_qp x n model types for joint
        each layer in 3rd dimension is model type
        first layer: vanilla vae
        second layer: beta vae
        third layer: rosetta VAE s
'''
