#!/usr/bin/env python
# coding: utf-8

# ## Deep Video Priors for Snapshot Compressive Imaging
#  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, MIT CSAIL"), [MIT CSAIL](https://www.csail.mit.edu/), yliu@csail.mit.edu, updated Dec 9, 2019.

# In[1]:


import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from dvp_linear_inv import (gap_denoise_bayer, 
                            gap_denoise, admm_denoise,
                            GAP_TV_rec, ADMM_TV_rec)

from utils import (A_, At_, psnr)


# In[2]:


# [0] environment configuration
# datasetdir = './dataset/original_dataset' # dataset            
# datasetdir = './dataset/benchmark/binary_mask_256_10f/bm_256_10f' # dataset
# datasetdir = './dataset/test/binary_mask_512_10f/bm_rescale_512_10f' # dataset
# datasetdir = './dataset/test/combine_binary_mask_512_10f/bm_rescale_512_10f' # dataset
# orig_dir = './dataset' # orig dataset
orig_dir = './dataset/benchmark/orig/bm_256_10f' # orig ataset
# mask_dir = './dataset' # orig dataset
mask_dir = './dataset/benchmark/mask' # mask dataset

resultsdir = './results' # results

# datname = 'test' # name of the dataset
orig_name = 'traffic'
# datname = 'kobe'


mask_name = 'binary_mask_256_10f'
# mask_name = 'traffic'

origpath = orig_dir + '/' + orig_name + '.mat' # path of the .mat orig file
maskpath = mask_dir + '/' + mask_name + '.mat' # path of the .mat mask file

# In[3]:


# [1] load orig and mask, calc meas
mat_v73_flag = 0  # v7.3 version .mat file flag

# load mask, orig
if not mat_v73_flag:
    origfile = sio.loadmat(origpath) # for '-v7.2' and below .mat file (MATLAB)
    maskfile = sio.loadmat(maskpath)
    orig = np.array(origfile['orig'])
    mask = np.array(maskfile['mask'])
    
    mask = np.float32(mask)
    orig = np.float32(orig)
    # print(mask.shape, orig.shape)
else:
    with h5py.File(origpath, 'r') as origfile: # for '-v7.3' .mat file (MATLAB)
        orig = np.array(origfile['orig'])
        orig = np.float32(orig).transpose((2,1,0))
        
    with h5py.File(maskpath, 'r') as maskfile: # for '-v7.3' .mat file (MATLAB)
        # print(list(file.keys()))
        mask = np.array(maskfile['mask'])
        mask = np.float32(mask).transpose((2,1,0))    
    # print(mask.shape, meas.shape, orig.shape)

#  calc meas
nmask = mask.shape[2]
norig = orig.shape[2]
meas = np.zeros([orig.shape[0], orig.shape[1], norig//nmask])
# print(mask.shape, meas.shape, orig.shape)

for i in range(norig//nmask):
    tmp_orig = orig[:,:,i*nmask:(i+1)*nmask]
    meas[:,:,i] = np.sum(tmp_orig*mask, 2)


# normalize data
mask_max = np.max(mask) 
mask = mask/mask_max
meas = meas/mask_max

iframe = 0
MAXB = 255.
(nrows, ncols, nmask) = mask.shape
if len(meas.shape) >= 3:
    meas = np.squeeze(meas[:,:,iframe])/MAXB
else:
    meas = meas/MAXB
orig = orig[:,:,iframe*nmask:(iframe+1)*nmask]/MAXB


# In[4]:

'''
## [2.1] GAP-TV [for baseline reference]
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'tv' # total variation (TV)
iter_max = 40 # maximum number of iterations
tv_weight = 0.1 # TV denoising weight (larger for smoother but slower)
tv_iter_max = 5 # TV denoising maximum number of iterations each
begin_time = time.time()
vgaptv_bayer,psnr_gaptv,ssim_gaptv,psnrall_tv =             gap_denoise_bayer(meas_bayer, mask_bayer, _lambda, 
                              accelerate, denoiser, iter_max, 
                              tv_weight=tv_weight, 
                              tv_iter_max=tv_iter_max,
                              X_orig=orig_bayer)
end_time = time.time()
tgaptv = end_time - begin_time
print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
    denoiser.upper(), mean(psnr_gaptv), mean(ssim_gaptv), tgaptv))
'''

# In[9]:


import torch
from packages.ffdnet.models import FFDNet

## [2.5] GAP/ADMM-FFDNet
### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'ffdnet' # video non-local network 
noise_estimate = False # disable noise estimation for GAP
# sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
# iter_max = [10,10,10,10] # maximum number of iterations
# sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
# iter_max = [10,10,10,10] # maximum number of iterations
sigma    = [50/255,25/255] # pre-set noise standard deviation,original, for traffic
iter_max = [20,20] # maximum number of iterations,original, for traffic
# sigma    = [35/255,15/255,12/255,6/255] # pre-set noise standard deviation
# iter_max = [10,10,10,10] # maximum number of iterations
useGPU = True # use GPU

# pre-load the model for FFDNet image denoising
in_ch = 1
model_fn = 'packages/ffdnet/models/net_gray.pth'
# Absolute path to model file
# model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)

# Create model
net = FFDNet(num_input_channels=in_ch)
# Load saved weights
if useGPU:
    state_dict = torch.load(model_fn)
    device_ids = [0]
    model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
else:
    state_dict = torch.load(model_fn, map_location='cpu')
    # CPU mode: remove the DataParallel wrapper
    state_dict = remove_dataparallel_wrapper(state_dict)
    model = net
model.load_state_dict(state_dict)
model.eval() # evaluation mode

begin_time = time.time()
vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_ffdnet =                 gap_denoise_bayer(meas, mask, _lambda, 
                                  accelerate, denoiser, iter_max, 
                                  noise_estimate, sigma,
                                  x0_bayer=None,
                                  X_orig=orig,
                                  model=model)
end_time = time.time()
tgapffdnet = end_time - begin_time
print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f} running time {:.1f} seconds.'.format(
    denoiser.upper(), mean(psnr_gapffdnet), mean(ssim_gapffdnet), tgapffdnet))


# In[11]:


import torch
from packages.fastdvdnet.models import FastDVDnet

## [2.2] GAP-FastDVDnet
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'fastdvdnet' # video non-local network 
noise_estimate = False # disable noise estimation for GAP
# sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
# iter_max = [10, 10, 10, 10, 10] # maximum number of iterations
sigma    = [50/255, 25/255, 12/255] # pre-set noise standard deviation,original, for traffic
iter_max = [20, 20, 25] # maximum number of iterations,original, for traffic
# sigma    = [50/255,25/255,12/255,6/255] # pre-set noise standard deviation
# iter_max = [10,10,10,10] # maximum number of iterations
useGPU = True # use GPU

# pre-load the model for FFDNet image denoising
NUM_IN_FR_EXT = 5 # temporal size of patch
model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

# Load saved weights
state_temp_dict = torch.load('./packages/fastdvdnet/model.pth')
if useGPU:
    device_ids = [0]
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
else:
    # CPU mode: remove the DataParallel wrapper
    state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
model.load_state_dict(state_temp_dict)

# Sets the model in evaluation mode (e.g. it removes BN)
model.eval()

begin_time = time.time()
vgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_fastdvdnet =                 gap_denoise_bayer(meas, mask, _lambda, 
                                  accelerate, denoiser, iter_max, 
                                  noise_estimate, sigma,
                                  x0_bayer=None,
                                  X_orig=orig,
                                  model=model)
end_time = time.time()
tgapfastdvdnet = end_time - begin_time
print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
    denoiser.upper(), mean(psnr_gapfastdvdnet), mean(ssim_gapfastdvdnet), tgapfastdvdnet))
    

# In[13]:


# import cv2
# # import scipy.io as sio

# # file = scipy.io.loadmat(matfile) # for '-v7.2' and below .mat file (MATLAB)
# # X = list(file[varname])

# # [3.3] result demonstration of GAP-Denoise
# vgapdenoise = vgapfastdvdnet_bayer
# psnr_gapdenoise = psnrall_fastdvdnet
# # vgapdenoise = vgapvnlnet
# # psnr_gapdenoise = psnr_gapvnlnet
# # vgapdenoise = vadmmvnlnet
# # psnr_gapdenoise = psnr_admmvnlnet


# fig = plt.figure(figsize=(15, 5))
# for nt in range(nmask):
#     plt.subplot(2, 4, nt+1)
#     # plt.imshow(orig_bayer[:,:,nt], cmap=plt.cm.gray, vmin=0, vmax=1)
#     orig_rgb = cv2.cvtColor(np.uint8(orig_bayer[:,:,nt]*MAXB), cv2.COLOR_BAYER_BG2BGR)
#     plt.imshow(orig_rgb)
#     plt.axis('off')
#     plt.title('Ground truth: Frame #{0:d}'.format(nt+1), fontsize=12)
# plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0, top=1, left=0, right=1)
# plt.show()   

# # fig = plt.figure(1)
# # fig = plt.figure(figsize=(12, 7))
# fig = plt.figure(figsize=(15, 5))
# PSNR_rec = np.zeros(nmask)
# for nt in range(nmask):
#     PSNR_rec[nt] = psnr(vgapdenoise[:,:,nt], orig_bayer[:,:,nt])
#     # ax = fig.add_subplot(2,4,nt+1)
#     plt.subplot(2, 4, nt+1)
#     # plt.imshow(vgapdenoise[:,:,nt], cmap=plt.cm.gray, vmin=0, vmax=1)
#     gapdenoise_rgb = cv2.cvtColor(np.uint8(vgapdenoise[:,:,nt]*MAXB), cv2.COLOR_BAYER_BG2BGR)
#     plt.imshow(gapdenoise_rgb)
#     plt.axis('off')
#     plt.title('Frame #{0:d} ({1:2.2f} dB)'.format(nt+1,PSNR_rec[nt]), fontsize=12)
    
# print('Mean PSNR {:2.2f} dB.'.format(np.mean(PSNR_rec)))
# # plt.title('GAP-{} mean PSNR {:2.2f} dB'.format(denoiser.upper(),np.mean(PSNR_rec)))
# plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0, top=1, left=0, right=1)
# plt.show()   

# plt.figure(2)
# plt.plot(PSNR_rec)
# plt.show()

# plt.figure(3)
# plt.plot(psnr_gapdenoise)
# plt.show()

# GAP-Net
savedmatdir = resultsdir + '/savedmat/'
if not os.path.exists(savedmatdir):
    os.makedirs(savedmatdir)

sio.savemat('{}gap{}_{}{:d}_sigma{:d}.mat'.format(savedmatdir,'-Net',orig_name,nmask,int(sigma[-1]*MAXB)),
            {'vgapffdnet':vgapffdnet, 
             'orig':orig,
             'psnr_gapffdnet':psnr_gapffdnet,
             'ssim_gapffdnet':ssim_gapffdnet,
             'psnrall_ffdnet':psnrall_ffdnet,
             'tgapffdnet':tgapffdnet,
             'vgapfastdvdnet':vgapfastdvdnet, 
             'psnr_gapfastdvdnet':psnr_gapfastdvdnet,
             'ssim_gapfastdvdnet':ssim_gapfastdvdnet,
             'psnrall_fastdvdnet':psnrall_fastdvdnet,
             'tgapfastdvdnet':tgapfastdvdnet})

''' 
# GAT-TV and GAP-Net
savedmatdir = resultsdir + '/savedmat/'
if not os.path.exists(savedmatdir):
    os.makedirs(savedmatdir)
# sio.savemat('{}gap{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
#             {'vgapdenoise':vgapdenoise},{'psnr_gapdenoise':psnr_gapdenoise})
# sio.savemat('{}gap{}_{}{:d}_sigma{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask,int(sigma[-1]*MAXB)),
sio.savemat('{}gap{}_{}{:d}_sigma{:d}.mat'.format(savedmatdir,'-TV&Net',datname,nmask,int(sigma[-1]*MAXB)),
            {'vgaptv':vgaptv, 
             'psnr_gaptv':psnr_gaptv,
             'ssim_gaptv':ssim_gaptv,
             'psnrall_tv':psnrall_tv,
             'tgaptv':tgaptv,
             'vgapffdnet':vgapffdnet, 
             'psnr_gapffdnet':psnr_gapffdnet,
             'ssim_gapffdnet':ssim_gapffdnet,
             'psnrall_ffdnet':psnrall_ffdnet,
             'tgapffdnet':tgapffdnet,
             'vgapfastdvdnet':vgapfastdvdnet, 
             'psnr_gapfastdvdnet':psnr_gapfastdvdnet,
             'ssim_gapfastdvdnet':ssim_gapfastdvdnet,
             'psnrall_fastdvdnet':psnrall_fastdvdnet,
             'tgapfastdvdnet':tgapfastdvdnet})
# sio.savemat(savedmatdir+'gaptv'+'_'+datname+str(ColT)+'.mat',{'vgaptv':vgaptv})
#np.save('Traffic_cacti_T8.npy', f)
'''