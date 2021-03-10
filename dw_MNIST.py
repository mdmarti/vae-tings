"""
A Variational Autoencoder (VAE) for spectrogram data.

VAE References
--------------
.. [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
	arXiv preprint arXiv:1312.6114 (2013).

	`<https://arxiv.org/abs/1312.6114>`_


.. [2] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic
	backpropagation and approximate inference in deep generative models." arXiv
	preprint arXiv:1401.4082 (2014).

	`<https://arxiv.org/abs/1401.4082>`_
"""
__date__ = "November 2018 - November 2019"


import numpy as np
import os
import torch
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from ava.models.vae_dataset import SyllableDataset
from ava.plotting.grid_plot import grid_plot
#from torch.utils.data import Dataset, DataLoader


X_SHAPE = (28,28)
"""Processed spectrogram shape: ``[freq_bins, time_bins]``"""
X_DIM = np.prod(X_SHAPE)
"""Processed spectrogram dimension: ``freq_bins * time_bins``"""



class ScaleBlock(nn.Module):

	"""
	One scaleblock inputs:
	- in_dim : the input dimension
	- n_layers : the number of resBlocks in the ScaleBlock
	- res_h_dim : hidden residual dimension
	- conv : whether or not res blocks should be fc or conv
	"""

	def __init__(self, in_dim, n_layers, res_h_dim,conv = True):
		super(ScaleBlock, self).__init__()

		self.scale_block = nn.ModuleList(
				[ResidualLayer(in_dim, res_h_dim, conv)]*n_layers)

	def forward(self, x):
		for layer in self.scale_block:
			x = x + layer(x)

		return x



class ResidualLayer(nn.Module):
	"""
	One residual layer inputs:
	- in_dim : the input dimension
	- res_h_dim : the hidden dimension of the residual block
	- conv : boolean, are hidden layers conv or fc?
	"""

	def __init__(self, in_dim, res_h_dim, conv):
		super(ResidualLayer, self).__init__()
		if conv:
			self.res_block = nn.Sequential(
				nn.BatchNorm2d(in_dim),
				nn.ReLU(),
				nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
						stride=1, padding=1, bias=False),
				nn.BatchNorm2d(res_h_dim),
				nn.ReLU(True),
				nn.Conv2d(res_h_dim, in_dim, kernel_size=1,
						stride=1, bias=False)
			)
		else:
			self.res_block = nn.Sequential(
				nn.BatchNorm1d(in_dim),
				nn.ReLU(True),
				nn.Linear(in_dim, res_h_dim),
				nn.BatchNorm1d(res_h_dim),
				nn.ReLU(True),
				nn.Linear(res_h_dim, in_dim))

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


class dw_VAE(nn.Module):
	"""Variational Autoencoder class for single-channel images.

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
	The model is trained to maximize the standard ELBO objective:

	.. math:: \mathcal{L} = \mathbb{E}_{q(z|x)} log p(x,z) + \mathbb{H}[q(z|x)]

	where :math:`p(x,z) = p(z)p(x|z)` and :math:`\mathbb{H}` is differential
	entropy. The prior :math:`p(z)` is a unit spherical normal distribution. The
	conditional distribution :math:`p(x|z)` is set as a spherical normal
	distribution to prevent overfitting. The variational distribution,
	:math:`q(z|x)` is an approximately rank-1 multivariate normal distribution.
	Here, :math:`q(z|x)` and :math:`p(x|z)` are parameterized by neural
	networks. Gradients are passed through stochastic layers via the
	reparameterization trick, implemented by the PyTorch `rsample` method.

	The dimensions of the network are hard-coded for use with 128 x 128
	spectrograms. Although a desired latent dimension can be passed to
	`__init__`, the dimensions of the network limit the practical range of
	values roughly 8 to 64 dimensions. Fiddling with the image dimensions will
	require updating the parameters of the layers defined in `_build_network`.
	"""

	def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0,
		device_name="auto", stage = 1):
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
		super(dw_VAE, self).__init__()
		self.save_dir = save_dir
		self.lr = lr
		self.z_dim = z_dim

		self.stage = stage
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
		self.model_precision = torch.tensor([model_precision],requires_grad=True).to(self.device)

	def _build_network(self):
		"""Define all the network layers."""
		if self.stage == 1:
			self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride=1,padding=1) #28x28x1 -> 28x28x32
			self.scale1 = ScaleBlock(in_dim=32, res_h_dim=32,n_layers=2) #28x28x32 -> 28x28x32
			self.conv2 = nn.Conv2d(32, 64, kernel_size=4,stride=2) #28x28x32 -> 13x13x64
			self.scale2 = ScaleBlock(in_dim=64,res_h_dim=64, n_layers=2) # 13x13x64 -> 13x13x64
			self.conv3 = nn.Conv2d(64, 128, kernel_size=6,stride=2) # 13x13x64 -> 4x4x128
			self.scale3 = ScaleBlock(in_dim=128,res_h_dim=128, n_layers=2)

			self.scale4 = ScaleBlock(in_dim=2048, res_h_dim=512, n_layers=2, conv = False)
			#self.fc13 = nn.Linear(49,49)
			self.fc11 = nn.Linear(2048,self.z_dim)
			self.fc12 = nn.Linear(2048,self.z_dim)
			#self.fc23 = nn.Linear(49,self.z_dim)
		## Quantizer
		#self.v_quantization = VectorQuantizer(self.n_embeddings, self.z_dim, self.beta)

		# Decoder
			self.fc2 = nn.Linear(self.z_dim,2048) #64 -> 2048
			self.convt1 = nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1) #2x2x512 -> 4x4x256
			self.scale5 = ScaleBlock(in_dim=256, res_h_dim=256, n_layers=2)
			self.convt2 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1) #4x4x256 -> 7x7x128
			self.scale6 = ScaleBlock(in_dim=128, res_h_dim=128, n_layers=2)
			self.convt3 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1) # 7x7x128 -> 14x14x64
			self.scale7= ScaleBlock(in_dim=64, res_h_dim=64, n_layers=2)
			self.convt4 = nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)  #14x14x64 -> 28x28x32
			self.scale8= ScaleBlock(in_dim=32, res_h_dim=32, n_layers=2)

			self.conv4 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)

		elif self.stage == 2:
			# Encoder
			self.fc1 = nn.Linear(self.z_dim,2048)
			self.fc2 = nn.Linear(2048, 2048)

			self.fc3 = nn.Linear(2048, 2048)
			self.fc41 = nn.Linear(2048, self.z_dim)

			self.fc42 = nn.Linear(2048, self.z_dim)


			#self.fc33 = nn.Linear(self.z_dim*2, self.z_dim*2)
			#self.fc43 = nn.Linear(self.z_dim*2, self.z_dim)

			# Decoder
			self.fc5 = nn.Linear(self.z_dim,2048)
			self.fc6 = nn.Linear(2048,2048)
			self.fc7 = nn.Linear(2048,2048)
			self.fc8 = nn.Linear(2048,self.z_dim)


	def _get_layers(self):
		"""Return a dictionary mapping names to network layers."""
		if self.stage == 1:
			return {'fc11':self.fc11, 'fc12':self.fc12,
		 			'fc2':self.fc2,
					'conv1':self.conv1, 'conv2':self.conv2,
					'conv3':self.conv3, 'conv4':self.conv4,
					'convt1':self.convt1, 'convt2':self.convt2,
					'convt3':self.convt3, 'convt4':self.convt4,
					'scale1':self.scale1, 'scale2':self.scale2,
					'scale3':self.scale3, 'scale4':self.scale4,
					'scale5':self.scale5, 'scale6':self.scale6,
					'scale7':self.scale7, 'scale8':self.scale8}
		elif self.stage == 2:
			return {'fc1':self.fc1, 'fc2':self.fc2, 'fc3':self.fc3,'fc41':self.fc41,
					'fc42':self.fc42,
					'fc5':self.fc5, 'fc6':self.fc6,'fc7':self.fc7,'fc8':self.fc8}


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
		mu : torch.Tensor
			Posterior mean, with shape ``[batch_size, self.z_dim]``
		u : torch.Tensor
			Posterior covariance factor, as defined above. Shape:
			``[batch_size, self.z_dim]``
		d : torch.Tensor
			Posterior diagonal factor, as defined above. Shape:
			``[batch_size, self.z_dim]``
		"""
		#x = x.unsqueeze(1)
		#print(x.shape)
		if self.stage == 1:
			print('Encoding')
			x = self.conv1(x)
			print(x)
			x = self.scale1(x)
			print(x)
			x = self.conv2(x)
			print(x)
			x = self.scale2(x)
			x = self.conv3(x)
			x = self.scale3(x)
			x = x.view(-1, 2048)
			#print(x.shape)
			x = self.scale4(x)
			mu = self.fc11(x)
			print('Done Encoding')
			S = torch.exp(self.fc12(x)).unsqueeze(-1)
			#print(S.shape)
			#print(S.permute(0,2,1).shape)
			#print(torch.matmul(S.permute(0,1,2),S.permute(0,2,1)).shape)
		#print('mu: {}, u: {}, d: {}'.format(mu, u, d))
			return mu, S
		elif self.stage == 2:

			x = torch.squeeze(x)

			x = self.fc1(x)
			x = self.fc2(x)
			x = self.fc3(x)

			mu = self.fc41(mu)

			S = torch.exp(self.fc42(x)).unsqueeze(-1)

			return mu, S


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

		if self.stage == 1:
			z = self.fc2(z)
			z = z.view(-1,512,2,2)
			z = self.convt1(z)
			z = self.scale5(z)
			z = self.convt2(z)
			z = self.scale6(z)
			z = self.convt3(z)
			z = self.scale7(z)
			z = self.convt4(z)
			z = self.scale8(z)
			z = torch.sigmoid(self.conv4(z))

			return z.view(-1, X_DIM)

		elif self.stage == 2:

			z = self.fc5(z)
			z = self.fc6(z)
			z = self.fc7(z)
			z = torch.sigmoid(self.fc8(z))

			return z.unsqueeze(1)


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
		mu, S = self.encode(x)
		#print(vari)
		#print(mu)
		#print(mu)
		#print(S)
		latent_dist = LowRankMultivariateNormal(mu, S, torch.ones(mu.shape,device=self.device))
		print(latent_dist)
		z = latent_dist.rsample()
		#print(z)
		x_rec = self.decode(z)
		# E_{q(z|x)} p(z)
		elbo = -0.5 * (torch.sum(torch.pow(z,2)) + self.z_dim * np.log(2*np.pi))
		# E_{q(z|x)} p(x|z)
		pxz_term = -0.5 * X_DIM * (torch.log(2*torch.tensor([np.pi],requires_grad=False).to(self.device)/self.model_precision))
		#print(pxz_term)
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
		#print(x_rec)
		#print(l2s)
		pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
		#print(pxz_term)
		elbo = elbo + pxz_term
		# H[q(z|x)]
		elbo = elbo + torch.sum(latent_dist.entropy())
		if return_latent_rec:
			return -elbo, z.detach().cpu().numpy(), \
				x_rec.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy(), -pxz_term
		return -elbo, -pxz_term, -torch.sum(latent_dist.entropy())


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
		train_rec = 0.0
		train_ent = 0.0

		for batch_idx, data in enumerate(train_loader):
			self.optimizer.zero_grad()
			#print(data.shape)
			(images, labels) = data
			images = images.to(self.device)
			#print(data)
			loss,rec,ent = self.forward(images)
			train_rec += rec.item()
			train_ent += ent.item()
			train_loss += loss.item()
			loss.backward()
			self.optimizer.step()
		train_loss /= len(train_loader.dataset)
		train_ent /= len(train_loader.dataset)
		train_rec /= len(train_loader.dataset)
		print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
				train_loss))
		print('Epoch: {} Average Rec Error: {:.4f}'.format(self.epoch, \
				train_rec))
		print('Epoch: {} Average Entropy: {:.4f}'.format(self.epoch, \
				train_ent))
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
		rec_err = 0.0
		recs = []
		with torch.no_grad():
			for i, data in enumerate(test_loader):
				(images, labels) = data
				images = images.to(self.device)
				loss,z,rec,rec_err_tmp = self.forward(images,return_latent_rec=True)
				recs.append(rec)
				test_loss += loss.item()
				rec_err += rec_err_tmp.item()

		test_loss /= len(test_loader.dataset)
		rec_err /= len(test_loader.dataset)
		print('Test loss: {:.4f}'.format(test_loss))
		print('Test reconstruction error: {:.4f}'.format(rec_err))
		return test_loss, rec_err, np.vstack(recs)


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
			if (vis_freq is not None) and (epoch % vis_freq == 0) and (self.stage == 1):
				self.visualize(loaders['test'])


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
		#print(filename)
		assert checkpoint['z_dim'] == self.z_dim
		layers = self._get_layers()
		#print(layers)
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
		# Retrieve spectrograms from the loader.
		#print(len(loader.dataset[indices]))
		#print(len(loader.dataset[indices][1]))
		#specs = torch.stack(torch.from_numpy(loader.dataset[indices][0])).to(self.device)
		(ims, labels) = loader.dataset[indices]
		specs = torch.from_numpy(np.array(ims[0])).to(self.device)
		specs = specs.unsqueeze(1)

		# Get resonstructions.
		with torch.no_grad():
			_, _, rec_specs,_ = self.forward(specs, return_latent_rec=True)
		specs = specs.squeeze(1).detach().cpu().numpy()
		#print(specs.shape)
		#print(rec_specs.shape)
		all_specs = np.stack([specs, rec_specs])
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
		latent = np.zeros((len(loader.dataset), self.z_dim))
		i = 0
		for data in loader:
			(images,labels) = data
			images = images.to(self.device)
			with torch.no_grad():
				mu, _ = self.encode(images)
			mu = mu.detach().cpu().numpy()
			latent[i:i+len(mu)] = mu
			i += len(mu)
		return latent



if __name__ == '__main__':
	pass


###
