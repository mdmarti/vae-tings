from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os
import copy

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE
from ava.models.vae import VAE
from ava.models.go_vae import goVAE
from ava.models.window_vae_dataset import get_window_partition, \
	get_fixed_window_data_loaders
from ava.preprocessing.preprocess import tune_window_preprocessing_params, \
    tune_syll_preprocessing_params, process_sylls
from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from ava.preprocessing.utils import get_spec
from ava.segmenting.segment import segment, tune_segmenting_params, get_audio_seg_filenames
from ava.segmenting.template_segmentation import get_template, segment_files
from ava.models.vae_dataset import get_syllable_partition, get_syllable_data_loaders

from sklearn.cluster import KMeans
import torch
import shutil


###########################
# -1) Functions useful for stuff later
###########################



#########################
# 0) Define Directories #
#########################

#root = 'path/to/parent/dir'
root = '/home/mmartinez/autoencoded-vocal-analysis/ava_test3'
trainAnimals = ['blu258']
goAnimals = ['blu288']
testAnimals = ['blu288','blu258']
trainDays = ['100']
testDays = ['100']
#testDays = ['100']

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

train_audio_dirs = [os.path.join(root,animal,'audio',day) for animal in trainAnimals for day in trainDays]
train_segment_dirs = [os.path.join(root,animal,'segs',day) for animal in trainAnimals for day in trainDays]
train_spec_dirs = [os.path.join(root,animal,'h5s',day) for animal in trainAnimals for day in trainDays]
train_proj_dirs = [os.path.join(root,animal,'proj',day) for animal in trainAnimals for day in trainDays]

model_filename = os.path.join(root,'model_258_0', 'checkpoint_100.tar')
plots_dir = os.path.join(root, 'plots_258_go_v3')

train_dc = DataContainer(projection_dirs=train_proj_dirs, audio_dirs=train_audio_dirs, \
    segment_dirs=train_segment_dirs,plots_dir=plots_dir, \
	spec_dirs=train_spec_dirs, model_filename=model_filename)

#####################################
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

segment_params = tune_segmenting_params(train_audio_dirs, segment_params)

#############################
# 1) Amplitude segmentation #
#############################

from_template = False #are we segmenting from template or doing amplitude seg?
debug = True
if debug:
	pass

else:
	if not from_template:
		for audio_dir, segment_dir in zip(train_audio_dirs, train_segment_dirs):
			if os.path.isdir(segment_dir):
				if len(os.listdir(segment_dir)) == 0: #if segmented files don't already exist
					segment(audio_dir, segment_dir, segment_params)
				else: continue
			else: segment(audio_dir, segment_dir, segment_params)



###########################################
# 1.5) Alternative: Template Segmentation #
###########################################
	else:
		template_dir = 'path/to/template/audio'
		template = get_template(template_dir, segment_params)
		result_train = segment_files(train_audio_dirs, train_segment_dirs, template, \
			segment_params, num_mad=8.0, njobs=8)
		result_test = segment_files(test_audio_dirs, test_segment_dirs, template, \
			segment_params, num_mad=8.0, njobs=8)

##################################################
# 2) Train a generative model on these syllables #
##################################################

	shotgun = False #are we doing shotgun VAE or syllable?



	preprocess_params = copy.copy(segment_params)
	preprocess_params["mel"] = False
	preprocess_params["time_stretch"] = True
	preprocess_params["real_preprocess_params"] = ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur')
	preprocess_params["int_preprocess_params"] = ('nperseg','noverlap')
	preprocess_params["binary_preprocess_params"]=("time_stretch","mel","within_syll_normalize")
	preprocess_params["sylls_per_file"] = 20
	preprocess_params["max_num_syllables"] = None

	preprocess_params = tune_syll_preprocessing_params(train_audio_dirs,train_segment_dirs, \
			preprocess_params)
	gen_train = zip(train_audio_dirs,train_segment_dirs,train_spec_dirs,repeat(preprocess_params))

	Parallel(n_jobs=4)(delayed(process_sylls)(*args) for args in gen_train)

if not os.path.isfile(model_filename):

	split = 1
	partition = get_syllable_partition(train_spec_dirs,split)
	num_workers = min(7,os.cpu_count()-1)
	train_loaders=get_syllable_data_loaders(partition, \
		num_workers=num_workers,batch_size=128)
	save_dir=os.path.join(root,'model_288')
	model=VAE(save_dir=save_dir)
	model.train_loop(train_loaders,epochs=101,test_freq=None)

else:

	split=1
	partition = get_syllable_partition(train_spec_dirs,split)
	num_workers = min(7,os.cpu_count()-1)

	train_loaders=get_syllable_partition(train_spec_dirs,split)
	num_workers = min(7,os.cpu_count()-1)
	train_loaders=get_syllable_data_loaders(partition, \
		num_workers=num_workers,batch_size=128)
	save_dir=os.path.join(root,'model_288_0')
	oldModel=VAE(save_dir=save_dir)
	oldModel.load_state(model_filename)



#test_partition = get_syllable_partition(test_spec_dirs,1)
#test_partition['test'] = test_partition['train']

#test_loaders = get_syllable_data_loaders(test_partition, \
#	num_workers=num_workers,batch_size=128)
#test_loaders['test'] = test_loaders['train']

# over-clustering latent means
lat_means_258 = train_dc.request('latent_means')
km_258_lat = KMeans(n_clusters=100).fit(lat_means_258)

km_258_centroids = km_258_lat.cluster_centers_
km_old_points = []
#print(lat_means_258[0,:])

for centroid_num, centroid in enumerate(km_258_centroids):
	cent_dists = np.linalg.norm(centroid.T - lat_means_258, axis=1)
	#min_dist_2 = np.sort(-cent_dists)
	min_dist = np.argmin(cent_dists)
	km_old_points.append(min_dist)

#print(km_old_points)
#print(min_dist_2[0])

def _is_audio_file(fn):
	return len(fn) >= 4 and fn[-4:] == '.wav'

def get_hdf5s_from_dir(dir):
	"""
	Return a sorted list of all hdf5s in a directory.

	.. warning:: ava.data.data_container relies on this.
	"""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
		_is_hdf5_file(f)]

def _is_hdf5_file(filename):
	"""Is the given filename an hdf5 file?"""
	return len(filename) > 5 and filename[-5:] == '.hdf5'


temp_filenames = [i for i in sorted(os.listdir(train_audio_dirs[0])) if \
		_is_audio_file(i)]
audio_filenames = [os.path.join(train_audio_dirs[0], i) for i in temp_filenames]
temp_filenames = [i[:-4] + '.txt' for i in temp_filenames]
seg_filenames = [os.path.join(train_segment_dirs[0], i) for i in temp_filenames]
spec_filenames = get_hdf5s_from_dir(train_spec_dirs[0])

#km_points_audio = []
#km_points_segs = []
#km_points_specs = []

#print(len(audio_filenames))
#print(len(seg_filenames))
#print(len(spec_filenames))
#for ind in km_old_points:
#	km_points_audio.append(audio_filenames[ind])
#	km_points_segs.append(seg_filenames[ind])
##	km_points_specs.append(spec_filenames[ind])

from ava.preprocessing.preprocess import get_audio_seg_filenames, read_onsets_offsets_from_file, get_syll_specs
import h5py

def make_old_spec_file(syll_inds, audio_dir,seg_dir, save_dir, p, shuffle=True, \
    verbose = True):
	if verbose:
		print("Creating old data spectrograms from", audio_dir)
		print("Getting segments from", seg_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	audio_filenames, seg_filenames = \
			get_audio_seg_filenames(audio_dir, seg_dir, p)
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(audio_filenames))
		np.random.seed(None)
		audio_filenames = np.array(audio_filenames)[perm]
		seg_filenames = np.array(seg_filenames)[perm]

	write_file_num = 0
	syll_data = {
		'specs':[],
		'onsets':[],
		'offsets':[],
		'audio_filenames':[],
	}
	sylls_per_file = p['sylls_per_file']
	syll_data_use = {
		'specs':[],
		'onsets':[],
		'offsets':[],
		'audio_filenames':[],
	}
	# For each pair of files...
	for audio_filename, seg_filename in zip(audio_filenames, seg_filenames):
		onsets, offsets = read_onsets_offsets_from_file(seg_filename, p)
		# Retrieve a spectrogram for each detected syllable.
		specs, good_sylls = get_syll_specs(onsets, offsets, audio_filename, p)
		onsets = [onsets[i] for i in good_sylls]
		offsets = [offsets[i] for i in good_sylls]
		# Add the syllables to <syll_data>.
		syll_data['specs'] += specs
		syll_data['onsets'] += onsets
		syll_data['offsets'] += offsets
		syll_data['audio_filenames'] += \
				len(onsets)*[os.path.split(audio_filename)[-1]]

	result = []
	for ind in syll_inds:
		#print(syll_data['specs'][ind])
		#print(syll_data['offsets'][ind])
		#print(syll_data['onsets'][ind])
		syll_data_use['specs'].append(syll_data['specs'][ind])
		syll_data_use['onsets'].append(syll_data['onsets'][ind])
		syll_data_use['offsets'].append(syll_data['offsets'][ind])
		syll_data_use['audio_filenames'].append(syll_data['audio_filenames'][ind])
		result.append(torch.from_numpy(syll_data['specs'][ind]).type(torch.FloatTensor))


	save_filename = \
		"syllables_old_100_" + str(write_file_num).zfill(4) + '.hdf5'
	save_filename = os.path.join(save_dir, save_filename)

	with h5py.File(save_filename, 'w') as f:

			# Add all the fields.
		for k in ['onsets', 'offsets']:
			f.create_dataset(k, \
					data=np.array(syll_data_use[k]))
		f.create_dataset('specs', \
			data=np.stack(syll_data_use['specs']))
		temp = [os.path.join(audio_dir, i) for i in \
				syll_data_use['audio_filenames']]
		f.create_dataset('audio_filenames', \
				data=np.array(temp).astype('S'))
		write_file_num += 1


	return torch.stack(result)


old_audio_dir = [os.path.join(root,animal,'old_mod_audio_centroids',day) for animal in trainAnimals for day in trainDays]
old_seg_dir = [os.path.join(root,animal,'old_mod_seg_centroids',day) for animal in trainAnimals for day in trainDays]
old_spec_dir = [os.path.join(root,animal,'old_mod_spec_centroids_v4',day) for animal in trainAnimals for day in trainDays]
old_proj_dir = [os.path.join(root,animal,'old_mod_proj_centroids',day) for animal in trainAnimals for day in trainDays]

'''
if not os.path.isdir(old_audio_dir):
	os.mkdir(old_audio_dir)

if not os.path.isdir(old_seg_dir):
	os.mkdir(old_seg_dir)
if not os.path.isdir(old_spec_dir[0]):
	os.mkdir(old_spec_dir)
'''
new_model_filename = os.path.join(root,'model_288_go_v3', 'checkpoint_100.tar')
new_plots_dir = os.path.join(root, 'plots_258_go_v3')

old_dc = DataContainer(projection_dirs=old_proj_dir, audio_dirs=train_audio_dirs, \
    segment_dirs=train_segment_dirs, spec_dirs=old_spec_dir,\
	plots_dir=new_plots_dir, model_filename=new_model_filename)

go_audio_dirs = [os.path.join(root,animal,'audio',day) for animal in goAnimals for day in trainDays]
go_segment_dirs = [os.path.join(root,animal,'segs',day) for animal in goAnimals for day in trainDays]
go_spec_dirs = [os.path.join(root,animal,'h5s',day) for animal in goAnimals for day in trainDays]
go_proj_dirs = [os.path.join(root,animal,'proj',day) for animal in goAnimals for day in trainDays]

go_dc = DataContainer(projection_dirs=go_proj_dirs, audio_dirs=go_audio_dirs, \
    segment_dirs=go_segment_dirs,plots_dir=new_plots_dir, model_filename=new_model_filename)

for audio_dir, segment_dir in zip(go_audio_dirs, go_segment_dirs):
	if os.path.isdir(segment_dir):
		if len(os.listdir(segment_dir)) == 0: #if segmented files don't already exist
			segment(audio_dir, segment_dir, segment_params)
		else: continue
	else: segment(audio_dir, segment_dir, segment_params)

preprocess_params = copy.copy(segment_params)
preprocess_params["mel"] = False
preprocess_params["time_stretch"] = True
preprocess_params["real_preprocess_params"] = ('min_freq', 'max_freq', 'spec_min_val', \
		'spec_max_val', 'max_dur')
preprocess_params["int_preprocess_params"] = ('nperseg','noverlap')
preprocess_params["binary_preprocess_params"]=("time_stretch","mel","within_syll_normalize")
preprocess_params["sylls_per_file"] = 20
preprocess_params["max_num_syllables"] = None

preprocess_params = tune_syll_preprocessing_params(train_audio_dirs,train_segment_dirs, \
		preprocess_params)

old_centroid_dir = os.path.join(root,'blu258','h5s_old','100')

old_data = make_old_spec_file(km_old_points, train_audio_dirs[0], train_segment_dirs[0], \
    old_centroid_dir, preprocess_params)


preprocess_params = tune_syll_preprocessing_params(go_audio_dirs,go_segment_dirs, \
		preprocess_params)

gen_go = zip(go_audio_dirs,go_segment_dirs,go_spec_dirs,repeat(preprocess_params))

Parallel(n_jobs=4)(delayed(process_sylls)(*args) for args in gen_go)
#used_mod = os.path.join(root,'model_288_beta_3.5','checkpoint_030.tar')

split = 0.25
partition = get_syllable_partition(go_spec_dirs,split)

num_workers = min(7,os.cpu_count()-1)
train_loaders=get_syllable_data_loaders(partition, \
	num_workers=num_workers,batch_size=128)
#train_loaders['test'] = train_loaders['train']
	#old_data=get_syllable_data_loaders(old_partition, \
	 #   num_workers=num_workers, batch_size=None)
#save_dir=os.path.join(root,'model_288_go_v3')
#maybe add in data discount?
#model=VAE(save_dir=save_dir)
with torch.no_grad():
	old_latents,_,_ = oldModel.encode(old_data)

beta_list = [3.5]#list(np.arange(0.5,3.6,0.75))

print(beta_list)
for ind, beta in enumerate(beta_list):
	print('Training Model Number ' + str(ind))
	print('Model Type: Beta = ' + str(beta))
	save_dir=os.path.join(root,'model_288_3_beta_' + str(beta))
	model=goVAE(save_dir=save_dir,use_old = 1, use_new = 1,beta=beta,zhat=True)
	#model.load_state(used_mod)
	model.train_loop(train_loaders,old_data,old_latents,epochs=101,test_freq=None)


'''
		print('Model Type: New data loss')

		save_dir=os.path.join(root,'model_288_new_loss')
		model=goVAE(save_dir=save_dir,use_old = 0, use_new = 1,beta=beta)
	elif train_type == 2:
		print('Model Type: Old data loss')

		save_dir=os.path.join(root,'model_288_old_loss')
		model=goVAE(save_dir=save_dir,use_old = 1, use_new = 0,beta=beta)
#model.load_state(used_mod)
#model.train_loop(train_loaders,epochs=101,test_freq=None)
	model.train_loop(train_loaders,old_data,old_latents,epochs=101,test_freq=None)
'''

'''
else:

	split=1
	partition = get_syllable_partition(train_spec_dirs,split)
	num_workers = min(7,os.cpu_count()-1)

	train_loaders=get_syllable_partition(train_spec_dirs,split)
	num_workers = min(7,os.cpu_count()-1)
	train_loaders=get_syllable_data_loaders(partition, \
		num_workers=num_workers,batch_size=128)
	save_dir=os.path.join(root,'model_288')
	model=VAE(save_dir=save_dir)
	model.load_state(model_filename)
'''

######################
# 3) Plot and analyze#
######################

def color_type(fName):
	if 'blu258' in fName:
		if '80' in fName:
			return '#A1057A'
		elif '100' in fName:
			return '#BF0D93'
		elif '115' in fName:
			return '#F00AB6'

	if 'blu288' in fName:
		if '80' in fName:
			return '#11F54A'
		elif '100' in fName:
			return '#59D478'
		elif '115' in fName:
			return '#43A35B'

'''
from ava.plotting.tooltip_plot import tooltip_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC
shotgun=False
if shotgun:
#	train_loaders['test'].dataset.write_hdf5_files(train_spec_dirs[0], num_files=1000)
#	test_loaders['test'].dataset.write_hdf5_files(test_spec_dirs[0], num_files=1000)


	tt_partition = get_window_partition(tt_audio_dirs, tt_segment_dirs, 1)
	tt_partition['test'] = tt_partition['train']
	tt_loaders = get_fixed_window_data_loaders(tt_partition,params, \
		num_workers=num_workers,batch_size=128)
'''
'''
temp_dc = DataContainer(projection_dirs=train_proj_dirs[:1], \
	audio_dirs=train_audio_dirs[:1], spec_dirs=train_spec_dirs[:1], plots_dir=plots_dir, \
	model_filename=model_filename)

temp_test_dc = DataContainer(projection_dirs=test_proj_dirs[:1], \
	audio_dirs=test_audio_dirs[:1], spec_dirs=test_spec_dirs[:1], plots_dir=plots_dir, \
	model_filename=model_filename)

tt_dc = DataContainer(projection_dirs=tt_proj_dirs,\
	audio_dirs=tt_audio_dirs, spec_dirs=tt_spec_dirs,plots_dir=plots_dir,\
	model_filename=model_filename)

latent_projection_plot_DC(tt_dc, alpha=0.25, s=0.5,filename='tt_latents_288_train_shotgun.pdf', color_by='filename_lambda',\
	condition_func= color_type)
'''
#latent_projection_plot_DC(temp_dc, alpha=0.25, s=0.5,filename='train_latents.pdf', color_by='filename_lambda',\
#	condition_func=color_type)
#tooltip_plot_DC(temp_dc, num_imgs=2000)

#latent_projection_plot_DC(temp_test_dc, alpha=0.25, s=0.5, filename='test_latents.pdf',color_by='filename_lambda',\
#	condition_func= color_type)
#tooltip_plot_DC(temp_test_dc, num_imgs=2000)


#For now, just need projections of each bird overlaid in latent space - colored by bird
