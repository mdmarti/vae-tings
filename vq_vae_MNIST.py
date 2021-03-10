"""
A Vector-Quantized Variational Autoencoder (vq-VAE) for spectrogram data.

VAE References
--------------
.. [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
	arXiv preprint arXiv:1312.6114 (2013).

	`<https://arxiv.org/abs/1312.6114>`_


.. [2] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic
	backpropagation and approximate inference in deep generative models." arXiv
	preprint arXiv:1401.4082 (2014).

	`<https://arxiv.org/abs/1401.4082>`_

	grouped up code from someone - check on that
"""

__date__ = "November 2018 - November 2019"


import numpy as np
import os
import torch
from torch.distributions import LowRankMultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
import umap

import matplotlib.pyplot as plt
import imageio

from ava.models.vae_dataset import SyllableDataset
from ava.plotting.grid_plot import grid_plot
#from vqvae.models.residual import ResidualStack
#from vqvae.models.quantizer import VectorQuantizer

X_SHAPE = (28,28)
"""Processed spectrogram shape: ``[freq_bins, time_bins]``"""
X_DIM = np.prod(X_SHAPE)
"""Processed spectrogram dimension: ``freq_bins * time_bins``"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################
#functions for vq-vae
##########################

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class DecodeResidualLayer(nn.Module):

    """
    One residual layer for decoding:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dim of residual res block
    """

    def __init__(self,out_dim,h_dim,res_h_dim):
        super(DecodeResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(h_dim,res_h_dim, kernel_size=3,
                               stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(res_h_dim, out_dim, kernel_size=1,
                               stride=1,bias=False)
		)
    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, decode=False):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers

        if decode:
            self.stack = nn.ModuleList(
                [DecodeResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)
        else:
            self.stack = nn.ModuleList(
                [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)


    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x



class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
	- this version will use a moving average to make latents
    """

    def __init__(self, n_e, e_dim, beta, gamma = 0.99):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.gamma = gamma

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e) #/ self.n_e, 1.0 / self.n_e)
        self.n_mat = torch.ones(1,self.n_e).to('cuda:0').detach()
        self.m_mat = torch.ones(self.n_e,self.e_dim).to('cuda:0').detach()

        self.n_mat_tmp = torch.ones(1,self.n_e).to('cuda:0').detach()
        self.m_mat_tmp = torch.ones(self.n_e,self.e_dim).to('cuda:0').detach()



    def forward(self, z,batch_size, n_train):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:
            0. Before quantizing: flatten output to (b size*H*W,C)
            1. get encoder input (B,z_dim)
            2. calculate distances
            3. after quantizing, reshape to b, c, h, w

        """

        z = z.contiguous()# z.permute(0, 2, 3, 1).contiguous()
        #print(self.embedding.weight.shape)
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)


        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        self.n_mat_tmp[:] = torch.sum(min_encodings, dim=0)
        self.n_mat = self.n_mat * self.gamma + self.n_mat_tmp * (1 - self.gamma)
        #print(z_flattened.shape)
        #print(self.m_mat.shape)
        #print(z_flattened.device)
        #print(min_encodings.device)
        #print(self.embedding.device)

        for ii in range(self.n_e):
            #print(self.m_mat.device)
            self.m_mat[ii,:] = self.gamma*self.m_mat[ii,:] + \
                    torch.sum(z_flattened[(min_encoding_indices == ii).squeeze(), :], dim = 0)*(1 - self.gamma)
            #print(self.m_mat[ii,:].shape)
            #print(self.embedding.weight.data.shape)
            #print(self.embedding.weight.data[ii,:].shape)
            #print(self.n_mat.shape)
            self.embedding.weight.data[ii,:] = self.m_mat[ii,:]/self.n_mat[0,ii].squeeze()

        # compute loss for embedding
        commit_loss = self.beta * torch.mean((z_q.detach()-z)**2)#n_train/batch_size*
        quant_loss =  torch.mean((z_q - z.detach()) ** 2)


        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        #z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return quant_loss, commit_loss, z_q, perplexity, min_encodings, min_encoding_indices


#################################
# VQVAE full
#################################

class VQ_VAE(nn.Module):
	"""Vector-Quantized Variational Autoencoder class for single-channel images.

	Attributes
	----------
	save_dir : str, optional
		Directory where the model is saved. Defaults to ``''``.
	lr : float, optional
		Model learning rate. Defaults to ``1e-3``.
	z_dim : int, optional
		Latent dimension. Defaults to ``32``.
	model_precision : float, optional
		Precision of the observation model. Defaults to ``10.0``.
	device_name : {'cpu', 'cuda', 'auto'}, optional
		Name of device to train the model on. When ``'auto'`` is passed,
		``'cuda'`` is chosen if ``torch.cuda.is_available()``, otherwise
		``'cpu'`` is chosen. Defaults to ``'auto'``.

	Notes
	-----
	The model is trained to maximize the modified ELBO objective:

	.. math:: \mathcal{L} = log p(x|z_{q}(x)) + ||sg(z_{e}(x) - e||^2
	 						+ \Beta ||z_{e}(x) - sg(e)||^2

	where 'sg` is the stop-gradient operator, z_{q} is the vector-Quantized
	embedding of x, z_{e}(x) is the encoder output, e is the set of embedding
	points, and || is the euclidean norm. math:: \Beta is a hyperparameter
	meant to weight the learning of the embedding space with learning of
	the encoder.


	The dimensions of the network are hard-coded for use with 128 x 128
	spectrograms. Although a desired latent dimension can be passed to
	`__init__`, the dimensions of the network limit the practical range of
	values roughly 8 to 64 dimensions. Fiddling with the image dimensions will
	require updating the parameters of the layers defined in `_build_network`.
	"""

	def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0,
		device_name="auto",n_embeddings=10, beta=1, save_img_embedding_map=False, n_train = 10000):
		"""Construct a VAE.

		Parameters
		----------
		save_dir : str, optional
			Directory where the model is saved. Defaults to the current working
			directory.
		lr : float, optional
			Learning rate of the ADAM optimizer. Defaults to 1e-3.
		z_dim : int, optional
			Dimension of the latent space. Defaults to 32.
		model_precision : float, optional
			Precision of the noise model, p(x|z) = N(mu(z), \Lambda) where
			\Lambda = model_precision * I. Defaults to 10.0.
		device_name: str, optional
			Name of device to train the model on. Valid options are ["cpu",
			"cuda", "auto"]. "auto" will choose "cuda" if it is available.
			Defaults to "auto".

		Note
		----
		- The model is built before it's parameters can be loaded from a file.
			This means `self.z_dim` must match `z_dim` of the model being
			loaded.
		"""

		"""
		to implement:

		"""
		####
		super(VQ_VAE, self).__init__()
		self.save_dir = save_dir
		self.lr = lr

		self.z_dim = z_dim
		self.n_embeddings = n_embeddings
		self.beta = beta

		self.model_precision = model_precision

		self.kernel = 4
		self.stride = 2
		self.h_dim = 32
		self.res_h_dim = 16


		assert device_name != "cuda" or torch.cuda.is_available()
		if device_name == "auto":
			device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		if self.save_dir != '' and not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self._build_network()
		self.optimizer = Adam(self.parameters(), lr=self.lr)
		self.epoch = 0
		self.loss = {'train':{}, 'test':{}}
		self.to(self.device)


		self.umap_t = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, \
			metric='euclidean', random_state=42)
		self.kmeans_t = KMeans(n_clusters=n_embeddings)
		self.qps_for_gif=[]
		self.vqps_for_gif=[]

		self.batch_size = 100
		self.n_train = n_train

	def _build_network(self):
		"""Define all the network layers."""
		# Encoder
		self.conv1 = nn.Conv2d(1, 16, kernel_size=4,stride=2,padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4,stride=2,padding=1)
		self.res1 = ResidualStack(in_dim=32, h_dim=32, res_h_dim=64, n_res_layers=1)
		self.fc1 = nn.Linear(1568,224)
		self.fc2 = nn.Linear(224,self.z_dim)

		# Quantizer
		self.v_quantization = VectorQuantizer(self.n_embeddings, self.z_dim, self.beta)

		# Decoder
		self.fc3 = nn.Linear(self.z_dim,224)
		self.fc4 = nn.Linear(224,1568)
		self.res2 = ResidualStack(in_dim=32, h_dim=32, res_h_dim=64, n_res_layers=1)
		self.convt1 = nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1)
		self.convt2 = nn.ConvTranspose2d(16,1,kernel_size=4,stride=2,padding=1)
		#self.convt3 = nn.ConvTranspose2d(16,1, kernel_size=4,stride=2,padding=1)


	def _get_layers(self):
		"""Return a dictionary mapping names to network layers."""

		return {'fc1':self.fc1,
		 		'fc2':self.fc2,
				'conv1':self.conv1, 'conv2':self.conv2,
				'v_quant':self.v_quantization,
				'convt1':self.convt1, 'convt2':self.convt2,
				'res1':self.res1, 'res2':self.res2}


	def encode(self, x):
		"""
		Compute :math:`q(z|x)`.

		.. math:: q(z|x) = \mathcal{N}(\mu, \Sigma)
		.. math:: \Sigma = u u^{T} + \mathtt{diag}(d)

		where :math:`\mu`, :math:`u`, and :math:`d` are deterministic functions
		of `x` and :math:`\Sigma` denotes a covariance matrix.

		Parameters
		----------
		x : torch.Tensor
			The input images, with shape: ``[batch_size, height=128,
			width=128]``

		Returns
		-------
		x : torch.Tensor
			latent space transformation ``[batch_size, self.z_dim]``

		"""
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.res1(x))
		x = x.view(-1, 1568)
		#print(x.shape)
		mu = F.relu(self.fc1(x))
		mu = self.fc2(mu)

		#print(x.shape)
		#x = self.fc1(x)

		return mu


	def decode(self, z):
		"""
		Compute :math:`p(x|z)`.

		.. math:: p(x|z) = \mathcal{N}(\mu, \Lambda)

		.. math:: \Lambda = \mathtt{model\_precision} \cdot I

		where :math:`\mu` is a deterministic function of `z`, :math:`\Lambda` is
		a precision matrix, and :math:`I` is the identity matrix.

		Parameters
		----------
		z : torch.Tensor
			Batch of latent samples with shape ``[batch_size, self.z_dim]``

		Returns
		-------
		x : torch.Tensor
			Batch of means mu, described above. Shape: ``[batch_size,
			X_DIM=128*128]``
		"""
		#print(z.shape)
		#print(self.fc1.weight.shape)
		#z = self.fc2(z)
		z = F.relu(self.fc3(z))
		z = F.relu(self.fc4(z))

		z = z.view(-1,32,7,7)
		z = F.relu(self.res2(z))
		z = F.relu(self.convt1(z))
		z = self.convt2(z)
		return z.view(-1, X_DIM)


	def forward(self, x, return_latent_rec=False):
		"""
		Send `x` round trip and compute a loss.

		In more detail: Given `x`, compute :math:`q(z|x)` and sample:
		:math:`\hat{z} \sim q(z|x)` . Then compute :math:`\log p(x|\hat{z})`,
		the log-likelihood of `x`, the input, given :math:`\hat{z}`, the latent
		sample. We will also need the likelihood of :math:`\hat{z}` under the
		model's prior: :math:`p(\hat{z})`, and the entropy of the latent
		conditional distribution, :math:`\mathbb{H}[q(z|x)]` . ELBO can then be
		estimated as:

		.. math:: 1/N \sum_{i=1}^N \mathbb{E}_{\hat{z} \sim q(z|x_i)}
			\log p(x_i,\hat{z}) + \mathbb{H}[q(z|x_i)]

		where :math:`N` denotes the number of samples from the data distribution
		and the expectation is estimated using a single latent sample,
		:math:`\hat{z}`. In practice, the outer expectation is estimated using
		minibatches.

		Parameters
		----------
		x : torch.Tensor
			A batch of samples from the data distribution (spectrograms).
			Shape: ``[batch_size, height=128, width=128]``
		return_latent_rec : bool, optional
			Whether to return latent means and reconstructions. Defaults to
			``False``.

		Returns
		-------
		loss : torch.Tensor
			Negative ELBO times the batch size. Shape: ``[]``
		latent : numpy.ndarray, if `return_latent_rec`
			Latent means. Shape: ``[batch_size, self.z_dim]``
		reconstructions : numpy.ndarray, if `return_latent_rec`
			Reconstructed means. Shape: ``[batch_size, height=128, width=128]``
		"""
		z_e = self.encode(x)
		#print(z_e.shape)
		#print(x.shape)
		#print(z_e.shape)
		#embedding_loss, z_q, perplexity, _, _ = self.v_quantization(z_e)
		quant_loss, commit_loss, z_q,_,_,_ = self.v_quantization(z_e, self.batch_size, self.n_train)
		#print(z_q.shape)
		#print(z_q.shape)
		x_rec = self.decode(z_q)
		#print(x_rec.shape)
		# E_{q(z|x)} p(z)
		#recon_loss = torch.mean((x_rec - x.view(-1,X_DIM))**2) / torch.var(x) #this is wrong
		pxz_term = -0.5 * X_DIM * (np.log(2*np.pi/self.model_precision))
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
		#print(l2s)
		pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
		#pxz_term = self.n_train/self.batch_size*pxz_term
		loss = -pxz_term + commit_loss#+  commit_loss #quant_loss +
		if return_latent_rec:
			return loss, z_q.detach().cpu().numpy(), \
				x_rec.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
		return loss, -pxz_term, quant_loss, commit_loss

	def forward_no_qps(self,x, return_latent_rec=False):

		z_e = self.encode(x)
		#print(z_e.shape)
		#embedding_loss, z_q, perplexity, _, _ = self.v_quantization(z_e)
		quant_loss,commit_loss, z_q,_,_,_ = self.v_quantization(z_e,self.batch_size, self.n_train)
		#print(z_q.shape)
		x_rec = self.decode(z_e)
		#print(x_rec.shape)
		# E_{q(z|x)} p(z)
		#recon_loss = torch.mean((x_rec - x.view(-1,X_DIM))**2) / torch.var(x) #this is wrong
		pxz_term = -0.5 * X_DIM * (np.log(2*np.pi/self.model_precision))
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
		#print(l2s)
		pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
		embedding_loss = quant_loss + commit_loss
		loss = -pxz_term + commit_loss#+ quant_loss + commit_loss

		if return_latent_rec:
			return loss, z_q.detach().cpu().numpy(), \
					x_rec.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
		return loss, -pxz_term, embedding_loss


	def train_epoch(self, train_loader):
		"""
		Train the model for a single epoch.

		Parameters
		----------
		train_loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader for training set

		Returns
		-------
		elbo : float
			A biased estimate of the ELBO, estimated using samples from
			`train_loader`.
		"""
		self.train()
		train_loss = 0.0
		train_pxz = 0.0
		train_q = 0.0
		train_c = 0.0
		for batch_idx, data in enumerate(train_loader):
			self.optimizer.zero_grad()
			#print(data.shape)
			(images,labels) = data
			images = images.to(self.device)
			#print(data.shape)
			loss, pxz, q_loss, c_loss = self.forward(images)
			train_pxz += pxz.item()
			train_q += q_loss.item()
			train_c += c_loss.item()
			train_loss += loss.item()
			loss.backward()
			self.optimizer.step()
		train_loss /= len(train_loader.dataset)
		train_pxz /= len(train_loader.dataset)
		train_q /= len(train_loader.dataset)
		train_c /= len(train_loader.dataset)

		print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
				train_loss))
		print('Epoch: {} Average rec err: {:.4f}'.format(self.epoch, \
				train_pxz))
		print('Epoch: {} Average quant loss: {:.4f}'.format(self.epoch, \
				train_q))
		print('Epoch: {} Average commit loss: {:.4f}'.format(self.epoch, \
				train_c))
		self.epoch += 1
		return train_loss


	def test_epoch(self, test_loader):
		"""
		Test the model on a held-out validation set, return an ELBO estimate.

		Parameters
		----------
		test_loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader for test set

		Returns
		-------
		elbo : float
			An unbiased estimate of the ELBO, estimated using samples from
			`test_loader`.
		"""
		self.eval()
		test_loss = 0.0
		with torch.no_grad():
			for i, data in enumerate(test_loader):
				data = data.to(self.device)
				loss,_,_ = self.forward(data)
				test_loss += loss.item()
		test_loss /= len(test_loader.dataset)
		print('Test loss: {:.4f}'.format(test_loss))
		return test_loss


	def train_loop(self, loaders, epochs=100, test_freq=2, save_freq=10,
		vis_freq=1):
		"""
		Train the model for multiple epochs, testing and saving along the way.

		Parameters
		----------
		loaders : dictionary
			Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
			torch.utils.data.Dataloader objects.
		epochs : int, optional
			Number of (possibly additional) epochs to train the model for.
			Defaults to ``100``.
		test_freq : int, optional
			Testing is performed every `test_freq` epochs. Defaults to ``2``.
		save_freq : int, optional
			The model is saved every `save_freq` epochs. Defaults to ``10``.
		vis_freq : int, optional
			Syllable reconstructions are plotted every `vis_freq` epochs.
			Defaults to ``1``.
		"""
		print("="*40)
		print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
		print("Training set:", len(loaders['train'].dataset))
		print("Test set:", len(loaders['test'].dataset))
		print("="*40)
		iter = 0
		# For some number of epochs...
		for epoch in range(self.epoch, self.epoch+epochs):
			# Run through the training data and record a loss.
			loss = self.train_epoch(loaders['train'])
			self.loss['train'][epoch] = loss
			# Run through the test data and record a loss.
			if (test_freq is not None) and (epoch % test_freq == 0):
				loss = self.test_epoch(loaders['test'])
				self.loss['test'][epoch] = loss
			# Save the model.
			if (save_freq is not None) and (epoch % save_freq == 0) and \
					(epoch > 0):
				filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
				self.save_state(filename)
			# Visualize reconstructions.
			if (vis_freq is not None) and (epoch % vis_freq == 0):
				embed = self.umap_t.fit_transform(self.get_qps())
				self.vqps_for_gif.append(embed)

				#print(len(self.vqps_for_gif))

				#if len(self.vqps_for_gif) > 1:
				#	print(self.vqps_for_gif[iter])
				#	iter += 1
				latents = self.get_latent(loaders['train'])
				print('clustering')
				kmeans = self.kmeans_t.fit(latents)
				centroids= kmeans.cluster_centers_
				#print(centroids)
				embed = self.umap_t.fit_transform(centroids)
				self.qps_for_gif.append(embed)

				self.visualize(loaders['test'])

		self.get_gif(vis_freq)


	def get_gif(self,vis_freq):

		#print(self.qps_for_gif[2])
		fnames = []
		for epoch_n, data in enumerate(self.qps_for_gif):
			ax = plt.gca()

			filename = os.path.join(self.save_dir,'qps_epoch{}.png'.format(vis_freq*epoch_n))
			fnames.append(filename)
			tmp_vqs = self.vqps_for_gif[epoch_n]
		#	print(tmp_vqs)
		#	print(data)
			vqs = ax.scatter(tmp_vqs[:,0],tmp_vqs[:,1])
			a1 = ax.scatter(data[:,0],data[:,1])
			plt.legend([a1,vqs],['kmeans qps', 'vqvae qps'])
			plt.xlabel('UMAP Component 1')
			plt.ylabel('UMAP Component 2')
			plt.tight_layout()
			plt.axis('square')
			plt.savefig(filename)
			plt.close()

		with imageio.get_writer(os.path.join(self.save_dir,'qps.gif'), mode='I') as writer:
			for name in fnames:
				image = imageio.imread(name)
				writer.append_data(image)

		#for name in set(fnames):
		#	os.remove(name)

	def save_state(self, filename):
		"""Save all the model parameters to the given file."""
		layers = self._get_layers()
		state = {}
		for layer_name in layers:
			state[layer_name] = layers[layer_name].state_dict()
		state['optimizer_state'] = self.optimizer.state_dict()
		state['loss'] = self.loss
		state['z_dim'] = self.z_dim
		state['epoch'] = self.epoch
		state['lr'] = self.lr
		state['save_dir'] = self.save_dir
		filename = os.path.join(self.save_dir, filename)
		torch.save(state, filename)


	def load_state(self, filename):
		"""
		Load all the model parameters from the given ``.tar`` file.

		The ``.tar`` file should be written by `self.save_state`.

		Parameters
		----------
		filename : str
			File containing a model state.

		Note
		----
		- `self.lr`, `self.save_dir`, and `self.z_dim` are not loaded.
		"""
		checkpoint = torch.load(filename, map_location=self.device)
		assert checkpoint['z_dim'] == self.z_dim
		layers = self._get_layers()
		for layer_name in layers:
			layer = layers[layer_name]
			layer.load_state_dict(checkpoint[layer_name])
		self.optimizer.load_state_dict(checkpoint['optimizer_state'])
		self.loss = checkpoint['loss']
		self.epoch = checkpoint['epoch']


	def visualize(self, loader, num_specs=5, gap=(2,6), \
		save_filename='reconstruction.pdf'):
		"""
		Plot spectrograms and their reconstructions.

		Spectrograms are chosen at random from the Dataloader Dataset.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			Spectrogram Dataloader
		num_specs : int, optional
			Number of spectrogram pairs to plot. Defaults to ``5``.
		gap : int or tuple of two ints, optional
			The vertical and horizontal gap between images, in pixels. Defaults
			to ``(2,6)``.
		save_filename : str, optional
			Where to save the plot, relative to `self.save_dir`. Defaults to
			``'temp.pdf'``.

		Returns
		-------
		specs : numpy.ndarray
			Spectgorams from `loader`.
		rec_specs : numpy.ndarray
			Corresponding spectrogram reconstructions.
		"""
		# Collect random indices.
		assert num_specs <= len(loader.dataset) and num_specs >= 1
		indices = np.random.choice(np.arange(len(loader.dataset)),
			size=num_specs,replace=False)
		#print(indices)
		#print(np.shape(np.array(loader.dataset[0])))
		#print(np.shape(np.array(loader.dataset[0:5])))
		# Retrieve spectrograms from the loader.
		#print(len(loader.dataset[indices]))
		#print(np.array(loader.dataset[indices]).shape)
		#print(loader.dataset[indices][0].shape)
		#print(loader.dataset[indices][0])
		#print(indices)
		(ims, labels) = loader.dataset[indices]
		#print(np.array(ims).shape)
		specs = torch.from_numpy(np.array(ims[0])).to(self.device)
		#print(specs.shape)
		# Get resonstructions.
		with torch.no_grad():
			_, _, rec_specs = self.forward_no_qps(specs.unsqueeze(1), return_latent_rec=True)
		specs = specs.detach().cpu().numpy()
		all_specs = np.stack([specs, rec_specs])
		print(all_specs.shape)
		# Plot.
		save_filename = os.path.join(self.save_dir, save_filename)
		grid_plot(all_specs, gap=gap, filename=save_filename)
		return specs, rec_specs


	def get_latent(self, loader):
		"""
		Get latent means for all syllable in the given loader.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader.

		Returns
		-------
		latent : numpy.ndarray
			Latent means. Shape: ``[len(loader.dataset), self.z_dim]``

		Note
		----
		- Make sure your loader is not set to shuffle if you're going to match
		  these with labels or other fields later.
		"""
		latent = np.zeros((len(loader.dataset), 64))
		i = 0
		for data in loader:
			(images, labels) = data
			images = images.to(self.device)
			with torch.no_grad():
				z_q = self.encode(images)
			#print(z_q.shape)
			z_q = z_q.detach().cpu().numpy()
			latent[i:i+len(z_q)] = z_q.reshape(-1,64)
			i += len(z_q)
		return latent

	def get_qps(self):
		"""
		Get quantization points in latent space

		Returns
		-----------
		latent : numpy.ndarray
		    Latent quantization points. Shape: ``[n_embeddings, self.z_dim]``

	    Note
		-------
		Nothing yet
		"""
		qps =  self.v_quantization.embedding.weight
		print(qps)
		return qps.detach().cpu().numpy()

	def vis_qp_recs(self):

		qps =  self.v_quantization.embedding.weight
		with torch.no_grad():
			qp_recs = self.decode(qps)
		qp_recs = qp_recs.view(-1,28,28).detach().cpu().numpy()
		mult = qp_recs.shape[0]/8
		for ii in range(8):
			save_filename = os.path.join(self.save_dir, 'latents_vis_qps_' + str(ii*mult) + 'to_' + str((ii+1)*mult - 1) + '.pdf')
			gap=(2,6)
			specs = np.stack([qp_recs[ii*mult:ii*mult + mult/2,:,:],qp_recs[ii*mult + mult/2:(ii+1)*mult,:,:]])
			print(specs.shape)
			grid_plot(specs, gap=gap, filename=save_filename)

if __name__ == '__main__':
	pass


###
