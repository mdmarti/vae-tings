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

import matplotlib.pyplot as plt


from ava.models.vae_dataset import SyllableDataset
from ava.plotting.grid_plot import grid_plot
#from vqvae.models.residual import ResidualStack
#from vqvae.models.quantizer import VectorQuantizer

X_SHAPE = (5,1)
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


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
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
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,z_dim): z_e(x)
            2. calculate distances
			3. create one-hot matrix, indicating which qp the sample is closest to
			4. create embedding matrix, return loss & qps

        """

        #### Initialize Tensors #####
        b_size = z.shape[0]
        ohMat = torch.zeros(b_size,self.n_e)

		# find closest qps
        for ind in range(b_size):
            sample = z[ind,:]
            dists = torch.linalg.norm(sample - self.embedding.weight,dim = 1)
            minInd = torch.argmin(dists)
            ohMat[ind,minInd] = 1

		# create z_q matrix

        z_q = torch.matmul(ohMat, self.embedding.weight)

		# create loss
        embed_loss = torch.sum(torch.pow(torch.linalg.norm(z.detach() - z_q, dim=1),2))
        commit_loss = self.beta * torch.sum(torch.pow(torch.linalg.norm(z - z_q.detach(), dim=1),2))

        loss = embed_loss + commit_loss
        #z_q = z + (z_q - z).detach()

        return loss, z_q #, perplexity, min_encodings, min_encoding_indices



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
		device_name="auto",n_embeddings=10, beta=1, save_img_embedding_map=False):
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
		Clean VectorQuantizer
		Clean Residual Stack

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


	def _build_network(self):
		"""Define all the network layers."""
		# Encoder
		self.fc1 = nn.Linear(5,16)
		self.fc2 = nn.Linear(16,32)
		self.fc3 = nn.Linear(32,16)
		self.fc4 = nn.Linear(16,self.z_dim)

		# Quantizer
		self.v_quantization = VectorQuantizer(self.n_embeddings, self.z_dim, self.beta)

		# Decoder
		self.fc5 = nn.Linear(self.z_dim,16)
		self.fc6 = nn.Linear(16,32)
		self.fc7 = nn.Linear(32,16)
		self.fc8 = nn.Linear(16,5)


	def _get_layers(self):
		"""Return a dictionary mapping names to network layers."""
		return {'fc1':self.fc1, 'fc2':self.fc2,
				'fc3':self.fc3, 'fc4':self.fc4,
				'fc5':self.fc5, 'fc6':self.fc6,
				'fc7':self.fc7, 'fc8':self.fc8,
				'v_quant':self.v_quantization}


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
		#x = x.unsqueeze(1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)


		return x


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
		z = F.relu(self.fc5(z))
		#print(z.shape)
		z = F.relu(self.fc6(z))
		z = F.relu(self.fc7(z))
		z = self.fc8(z)

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

		embedding_loss, z_q = self.v_quantization(z_e)

		x_rec = self.decode(z_q)

		#recon_loss = torch.mean((x_rec - x.view(-1,X_DIM))**2) / torch.var(x) #this is wrong
		pxz_term = -0.5 * X_DIM * (np.log(2*np.pi/self.model_precision))
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
		#print(l2s)
		pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)

		#loss = torch.sum(l2s) + embedding_loss
		loss = -pxz_term + embedding_loss
		if return_latent_rec:
			return loss, z_q.detach().cpu().numpy(), \
				x_rec.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
		return loss, -pxz_term, embedding_loss

	def forward_no_qps(self,x, return_latent_rec=False):

		z_e = self.encode(x)
		#print(z_e.shape)
		#embedding_loss, z_q, perplexity, _, _ = self.v_quantization(z_e)
		embedding_loss, z_q = self.v_quantization(z_e)
		#print(z_q.shape)
		x_rec = self.decode(z_q)
		#print(x_rec.shape)
		# E_{q(z|x)} p(z)
		#recon_loss = torch.mean((x_rec - x.view(-1,X_DIM))**2) / torch.var(x) #this is wrong

		pxz_term = -0.5 * X_DIM * (np.log(2*np.pi/self.model_precision))
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
		#print(l2s)
		pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)

		loss = torch.sum(l2s) + embedding_loss

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
		train_embed = 0.0
		for batch_idx, data in enumerate(train_loader):
			self.optimizer.zero_grad()
			gauss, labels = data
			#print(data.shape)
			#print(labels.shape)
			gauss = gauss.to(self.device).squeeze(dim=1)
			#print(gauss.shape)
			#print(data.shape)
			loss, pxz, embed_L = self.forward(gauss)
			train_pxz += pxz.item()
			train_embed += embed_L.item()
			train_loss += loss.item()
			#pxz.backward(retain_graph=True)
			#embed_L.backward()
			loss.backward()
			self.optimizer.step()
		train_loss /= len(train_loader.dataset)
		train_pxz /= len(train_loader.dataset)
		train_embed /= len(train_loader.dataset)

		print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
				train_loss))
		print('Epoch: {} Average rec err: {:.4f}'.format(self.epoch, \
				train_pxz))
		print('Epoch: {} Average embed loss: {:.4f}'.format(self.epoch, \
				train_embed))
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
				(gauss,label) = data
				gauss = gauss.to(self.device)
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
		#print("Test set:", len(loaders['test'].dataset))
		print("="*40)
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
				self.visualize(loaders['train'])


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


	def visualize(self, loader, num_specs=200, gap=(2,6), \
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

		(gauss,labels) = loader.dataset[indices]
		gauss = torch.cat(gauss).to(self.device)
		labels = np.concatenate(labels,axis=0)
		unique_labels = np.unique(labels)
		#print(specs.shape)
		# Get resonstructions.
		with torch.no_grad():
			_, _, rec_specs = self.forward_no_qps(gauss, return_latent_rec=True)
		specs = gauss.detach().cpu().numpy()
		#all_specs = np.stack([specs, rec_specs])
		# Plot.
		ax = plt.gca()

		lines = []
		colors = ['black','firebrick','tomato','sienna','chartreuse','darkgreen','darkorchid','darkblue']
		for idx,label in enumerate(unique_labels):
			tmp1 = ax.scatter(specs[labels==label,0],specs[labels==label,1],marker='x',color=colors[idx])
			tmp2 = ax.scatter(rec_specs[labels==label,0],specs[labels==label,1],marker='o',color=colors[idx])
			lines.append(tmp1)

		save_filename = os.path.join(self.save_dir, save_filename)

		plt.legend(lines,['g1','g2','g3','g4','g5','g6','g7','g8'])
		plt.xlabel('Dim 1')
		plt.ylabel('Dim 2')
		plt.tight_layout()
		plt.axis('square')
		plt.savefig(save_filename)
		plt.close('all')
		#grid_plot(all_specs, gap=gap, filename=save_filename)
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
		latent = np.zeros((len(loader.dataset), self.z_dim))
		latent_labels = np.zeros((len(loader.dataset),1))
		i = 0
		for data in loader:
			(gauss,labels) = data
			gauss = gauss.to(self.device).squeeze(1)#torch.cat(gauss).to(self.device)
			#data = data.to(self.device)
			with torch.no_grad():
				z_q = self.encode(gauss)
			z_q = z_q.detach().cpu().numpy()
			latent[i:i+len(z_q)] = z_q
			latent_labels[i:i+len(z_q)] = labels
			i += len(z_q)
		return latent, latent_labels

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

if __name__ == '__main__':
	pass


###
