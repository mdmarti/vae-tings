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

""" TO DO

get this to work lol implement actual vqvae loss function

"""
__date__ = "November 2018 - November 2019"


import numpy as np
import os
import torch
from torch.distributions import LowRankMultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from ava.models.vae_dataset import SyllableDataset
from ava.plotting.grid_plot import grid_plot


X_SHAPE = (128,128)
"""Processed spectrogram shape: ``[freq_bins, time_bins]``"""
X_DIM = np.prod(X_SHAPE)
"""Processed spectrogram dimension: ``freq_bins * time_bins``"""
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





class QP_VAE(nn.Module):
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

	def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0, tau=2, \
	            use_old=1,use_new=1, no_sample=True, old_latent=[], device_name="auto"):
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
		super(QP_VAE, self).__init__()
		self.no_sample = no_sample
		self.Tau = tau
		self.old_latent = old_latent
		self.use_old = use_old
		self.use_new = use_new
		self.save_dir = save_dir
		self.lr = lr
		self.z_dim = z_dim
		self.model_precision = model_precision
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
		self.conv1 = nn.Conv2d(1,12, kernel_size=4,stride=2,padding=1) #out: 64x64x12
		self.conv2 = nn.Conv2d(12, 24, kernel_size=4,stride=2,padding=1) #out: 32x32x24
		self.res1 = ResidualStack(in_dim=24, h_dim=24, res_h_dim=48, n_res_layers=2) #out: 32x32x24
		self.conv3 = nn.Conv2d(24, 48, kernel_size=4,stride=2,padding=1) #out: 16x16x48
		self.conv4 = nn.Conv2d(48, 96, kernel_size=4,stride=2, padding=1) #out: 8x8x96
		self.res2 = ResidualStack(in_dim=96, h_dim=96, res_h_dim=192, n_res_layers=2) #out: 8x8x96
	#	self.conv5 = nn.Conv2d(48, 96, kernel_size=3,stride=1,padding=1) #out: 16x16x96
	#	self.conv6 = nn.Conv2d(96, 96, kernel_size=4,stride=2, padding=1) #out: 8x8x96
		self.fc1 = nn.Linear(6144,1568)
		self.fc21 = nn.Linear(1568,224)
		self.fc22 = nn.Linear(1568,224)
		self.fc23 = nn.Linear(1568,224)
		self.fc31 = nn.Linear(224,self.z_dim)
		self.fc32 = nn.Linear(224,self.z_dim)
		self.fc33 = nn.Linear(224,self.z_dim)
		# Decoder
		self.fc4 = nn.Linear(self.z_dim,224)
		self.fc5 = nn.Linear(224,1568)
		self.fc6 = nn.Linear(1568,6144)
	#	self.convt1 = nn.ConvTranspose2d(96,96,kernel_size=4,stride=2,padding=1) #out:
	#	self.convt2 = nn.ConvTranspose2d(96,48,kernel_size=3,stride=1,padding=1) #out:
		self.res3 = ResidualStack(in_dim=96, h_dim=96, res_h_dim=192, n_res_layers=2)
		self.convt1 = nn.ConvTranspose2d(96,48,kernel_size=4,stride=2,padding=1) #out: 16x16x96
		self.convt2 = nn.ConvTranspose2d(48,24,kernel_size=4,stride=2,padding=1) #out: 32x32x48
		self.res4 = ResidualStack(in_dim=24, h_dim=24, res_h_dim=48, n_res_layers=2)
		self.convt3 = nn.ConvTranspose2d(24,12,kernel_size=4,stride=2,padding=1) #out: 64x64x24
		self.convt4 = nn.ConvTranspose2d(12,1,kernel_size=4,stride=2,padding=1)


	def _get_layers(self):
		"""Return a dictionary mapping names to network layers."""
		return {'fc1':self.fc1,
				'fc21':self.fc21, 'fc22':self.fc22, 'fc23':self.fc23,
				'fc31':self.fc31, 'fc32':self.fc32, 'fc33':self.fc33,
				'fc4':self.fc4, 'fc5':self.fc5, 'fc6': self.fc6,
				'conv1':self.conv1, 'conv2':self.conv2,
				'conv3':self.conv3, 'conv4':self.conv4,
				#'conv5':self.conv5, 'conv6':self.conv6,
				'convt1':self.convt1, 'convt2':self.convt2,
				'convt3':self.convt3, 'convt4':self.convt4,
				#'convt5':self.convt5, 'convt6':self.convt6,
				'res1':self.res1, 'res2':self.res2,
				'res3':self.res3,'res4':self.res4}


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
#		x = x.permute(0,3,1,2)
		x = x.unsqueeze(1)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.res1(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.res2(x))
		#x = F.relu(self.conv5(x))
		#x = F.relu(self.conv6(x))
		#print(x.shape)
		x = x.contiguous().view(-1, 6144)
		#print(x.shape)
		x = F.relu(self.fc1(x))
		mu = F.relu(self.fc21(x))
		mu = self.fc31(mu)

		u = F.relu(self.fc22(x))
		u = self.fc32(u).unsqueeze(-1) # Last dimension is rank of \Sigma = 1.
		d = F.relu(self.fc23(x))
		d = torch.exp(self.fc33(d)) # d must be positive.
		#print('mu: {}, u: {}, d: {}'.format(mu, u, d))
		#print(mu)
		return mu, u, d


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
		z = F.relu(self.fc4(z))
		z = F.relu(self.fc5(z))
		z = F.relu(self.fc6(z))

		z = z.reshape(-1,96,8,8)
		#z = F.relu(self.convt1(z))
		#z = F.relu(self.convt2(z))
		z = F.relu(self.res3(z))
		z = F.relu(self.convt1(z))
		z = F.relu(self.convt2(z))
		z = F.relu(self.res4(z))
		z = F.relu(self.convt3(z))
		z = self.convt4(z)
	#	z = F.relu(self.convt3(z))
#		z = self.conv4(z)
	#	print(z.shape)
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
		mu, u, d = self.encode(x)
		latent_dist = LowRankMultivariateNormal(mu, u, d)
		z = latent_dist.rsample()
		x_rec = self.decode(z)
		# E_{q(z|x)} p(z)
		elbo = -0.5 * (torch.sum(torch.pow(z,2)) + self.z_dim * np.log(2*np.pi))
		#print(elbo.shape)
		# E_{q(z|x)} p(x|z)
		pxz_term = -0.5 * X_DIM * (np.log(2*np.pi/self.model_precision))
		#print(pxz_term.shape)
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
		#print(l2s.shape)
		pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
		elbo = elbo + pxz_term
		# H[q(z|x)]
		ent = torch.sum(latent_dist.entropy())
		elbo = elbo + torch.sum(latent_dist.entropy())
		#print(elbo.shape)
		if return_latent_rec:
			return -elbo, z.detach().cpu().numpy(), \
				x_rec.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy(), -pxz_term
		return -elbo, -pxz_term, -ent

	def old_data_loss(self, x_o, z_o):

		"""
		Send old x semi-round-trip, compute a loss.
		this is only for old data points, from a previously trained model.

		In more detail: given '\hat{x}', a datapoint from an old/target distribution,
		compute MSE(x,\hat{x}). This is to say, compute the difference between our
		original datapoint and the reconstructions.
		Also computer MSE(\hat{z},z). The difference between our original latent points
		from the previous model and our new model latent points.

		Recall that MSE = b^2 + V, so reducing MSE will train our model to reduce both
		bias and variance. Therefore we might become 'overconfident' about the
		position of the latent points, in both the new and old datasets. Therefore it may
		be better to just train our model to reduce bias, and let the new data points
		work on specifying the variance bits of the network.

		This is what we're doing here and now. Just minimizing bias for latent points,
		but fully training the decoder

		"""
		#x_o = x_o.unsqueeze(1)
		z_mu_o, u_o, d_o = self.encode(x_o)
		latent_o = LowRankMultivariateNormal(z_mu_o, u_o, d_o)
		z_old_rec = latent_o.rsample()
		if self.no_sample:
			x_old_rec = self.decode(z_o)
		else:
			x_old_rec = self.decode(z_old_rec)
		#print(x_old_rec.shape)
        # E_{q(z_o|x_o)} p(z)}
		mse_x_new = torch.sum(torch.pow(x_o.view(x_o.shape[0],-1) - x_old_rec, 2))
		#print(elbo_old.shape)
		mse_z_new = torch.sum(torch.pow(z_o.view(z_o.shape[0],-1) - z_mu_o,2))
		mse = mse_x_new + mse_z_new
		#print(elbo_old.shape)
		return self.Tau * mse

	def train_epoch(self, train_loader, old_data, z_o):
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
		train_loss_old = 0.0
		old_data = old_data.to(self.device)
		z_o = z_o.to(self.device)

		rec_err = 0.0
		ent_all = 0.0

		for batch_idx, data in enumerate(train_loader):
			self.optimizer.zero_grad()
			data = data.to(self.device)
			loss, pxz, ent = self.forward(data)
			train_loss += loss.item()
			old_loss = self.old_data_loss(old_data,z_o)
			train_loss += loss.item()
			train_loss_old += old_loss.item()

			#loss.backward()
			if self.use_new:
				loss.backward()
			if self.use_old:
				old_loss.backward()
			self.optimizer.step()


		train_loss /= len(train_loader.dataset)
		train_loss_old /= (len(z_o)*(batch_idx + 1))
		#train_loss /= len(train_loader.dataset)

		print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
				train_loss))
		print('Epoch: {} Average loss, old data: {:.4f}'.format(self.epoch, \
				train_loss_old))
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
				loss = self.forward(data)
				test_loss += loss.item()
		test_loss /= len(test_loader.dataset)
		print('Test loss: {:.4f}'.format(test_loss))
		return test_loss


	def train_loop(self, loaders,old_data, old_latents, epochs=100, test_freq=2, save_freq=10,
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
			loss = self.train_epoch(loaders['train'],old_data,old_latents)
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
				self.visualize(loaders['test'],save_filename='qpset_reconstruction.pdf')
				self.visualize(loaders['train'],save_filename='trainset_reconstruction.pdf')


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
		# Retrieve spectrograms from the loader.
		specs = torch.stack(loader.dataset[indices]).to(self.device)
		# Get resonstructions.
		with torch.no_grad():
			_, _, rec_specs,_ = self.forward(specs, return_latent_rec=True)
		specs = specs.detach().cpu().numpy()
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
			data = data.to(self.device)
			with torch.no_grad():
				mu, _, _ = self.encode(data)
			mu = mu.detach().cpu().numpy()
			latent[i:i+len(mu)] = mu
			i += len(mu)
		return latent


if __name__ == '__main__':
	pass


###
