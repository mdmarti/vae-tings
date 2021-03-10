'''
MY TIME TO SHINE WITH THIS MMD PLOTTING
'''


from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os
from matplotlib.collections import PolyCollection
from matplotlib.colors import cnames
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE, MDS


def gauss_kernel(x1,x2, gamma=0.05):
    '''
    expects:
    - x1: matrix of x values, samples x dimensions
    - x2: matrix of x values, samples x dimensions
    - gamma: sd of gaussian kernel, essentially acts as regularizer
    '''
    return np.exp(-gamma * np.sum(np.pow(x1 - x2, 2)))

def estimate_median_sigma(latent, n=10000):
    """
    Estimate the median pairwise distance for use as a kernel bandwidth.

    Parameters
    ----------
    latent : numpy.ndarray
        Latent means.
    n : int, optional
        Number of random pairs to draw. Defaults to `10000`.

    Returns
    -------
    sigma : float
        Median pairwise euclidean distance between sampled latent means.
    """
    arr = np.zeros(n)
    for i in range(n):
        i1, i2 = np.random.randint(len(latent)), np.random.randint(len(latent))
        arr[i] = np.sum(np.power(latent[i1]-latent[i2],2))
    return np.median(arr)

def estimate_mmd_unbiased_quadratic(l_sample1, l_sample2, sigma = 0.05):

    m = len(l_sample1)
    n = len(l_sample2)
    t1 = np.zeros((m,n))
    t2 = np.zeros((m,n))
    t3 = np.zeros((m,n))
    for ii in range m:
        for jj in range n:
            if not(ii == jj):
                t1[ii,jj] = gauss_kernel(l_sample1[ii,:],l_sample1[jj,:],sigma)
                t2[ii,jj] = gauss_kernel(l_sample2[ii,:],l_sample2[jj,:],sigma)
            t3[ii,jj] = gauss_kernel(l_sample1[ii,:],l_sample2[jj,:],sigma)
    mmd2 = np.sum(t1)/(m*(m-1)) + np.sum(t2)/(n*(n-1)) - 2*np.sum(t3)/(m*n)

    return mmd2

def _mmd_helper()


if __name__ == '__main__':
    pass
