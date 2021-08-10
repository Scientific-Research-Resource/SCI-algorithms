# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# ## GAP-TV for Video Compressive Sensing
# ### GAP-TV
# > X. Yuan, "Generalized alternating projection based total variation minimization for compressive sensing," in *IEEE International Conference on Image Processing (ICIP)*, 2016, pp. 2539-2543.
# ### Code credit
# [Xin Yuan](https://www.bell-labs.com/usr/x.yuan "Dr. Xin Yuan, Bell Labs"), [Bell Labs](https://www.bell-labs.com/), xyuan@bell-labs.com, created Aug 7, 2018.  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, Tsinghua University"), [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), y-liu16@mails.tsinghua.edu.cn, updated Jan 20, 2019.

# %%
import os
import time
from datetime import datetime
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from pnp_sci_algo import admmdenoise_cacti
from joint_pnp_sci_algo import joint_admmdenoise_cacti

from utils import (A_, At_, show_n_save_res)
import torch
from packages.ffdnet.models import FFDNet
from packages.fastdvdnet.models import FastDVDnet

# %%
# [0] environment configuration
# GPU assign
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# datasetdir = './dataset/cacti' # dataset
datasetdir = './dataset/data'
# datasetdir = './dataset/benchmark/data/combine_binary_mask_256_10f/bm_256_10f'
# datasetdir = '../gapDenoise/dataset/cacti' # dataset
resultsdir = './results' # results


# datname = 'kobe32'        # name of the dataset
# datname = 'traffic48'     # name of the dataset
# datname = 'runner40'      # name of the dataset
# datname = 'drop40'        # name of the dataset
# datname = 'crash32'       # name of the dataset
# datname = 'aerial32'      # name of the dataset
# datname = 'bicycle24'     # name of the dataset
# datname = 'data_kobe'    # name of the dataset
datname = 'data_traffic_256_10f'    # name of the dataset

## flags and params
save_res_flag = 1          # save results
show_res_flag = 1           # show results
save_param_flag =  1        # save params
# choose algorithms: 'all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet', 'gapfastdvdnet', 'admmfastdvdnet','gaptv+fastdvdnet', 'admmtv+fastdvdnet'
# test_algo_flag = ['all' ]		
test_algo_flag = ['admmtv+fastdvdnet']


# matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file
matfile = datasetdir + '/' + datname + '.mat' # path of the .mat data file


# noise
gaussian_noise_level = 0
poisson_noise = False


# %%
# [1] load data
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version

if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
    order = 'K' # [order] keep as the default order in Python/numpy
    meas = np.float32(file['meas'], order=order)
    mask = np.float32(file['mask'], order=order)
    orig = np.float32(file['orig'], order=order)
else: # MATLAB .mat v7.3
    file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    order = 'F' # [order] switch to MATLAB array order
    meas = np.float32(file['meas'], order=order).transpose()
    mask = np.float32(file['mask'], order=order).transpose()
    orig = np.float32(file['orig'], order=order).transpose()

# print('meas, mask, orig:', meas.shape, mask.shape, orig.shape)

    
# zzh: expand dim for a single 'meas'
if meas.ndim<3:
    meas = np.expand_dims(meas,2)
    # print(meas.shape)
# print('meas, mask, orig:', meas.shape, mask.shape, orig.shape)

# add nosie
# print('* before add noise: orig {}  mask {} meas {}'.format(np.mean(orig), np.mean(mask), np.mean(meas)))
gaussian_noise = np.random.randn(meas.shape[0], meas.shape[1], meas.shape[2])*gaussian_noise_level
meas = meas + gaussian_noise
if poisson_noise:
    meas = np.random.poisson(meas)
 
# normalize data
mask_max = np.max(mask) 
mask = mask/mask_max
meas = meas/mask_max   

iframe = 0
nframe = 1
nmask = mask.shape[2]
MAXB = 255.

# common parameters and pre-calculation for PnP
# define forward model and its transpose
A  = lambda x :  A_(x, mask) # forward model function handle
At = lambda y : At_(y, mask) # transpose of forward model

# mask_sum = np.sum(mask, axis=2)
# mask_sum[mask_sum==0] = 1


# %%
## [2.1] GAP/ADMM-TV
### [2.1.1] GAP-TV
if ('all' in test_algo_flag) or ('gaptv' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv' # total variation (TV)
    iter_max = 100 # maximum number of iterations
    tv_weight = 0.25 # TV denoising weight (larger for smoother but slower) [kobe:0.25; ]
    tv_iter_max = 5 # TV denoising maximum number of iterations each

    vgaptv,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv = admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate,
                                            denoiser=denoiser, iter_max=iter_max, 
                                            tv_weight=tv_weight, 
                                            tv_iter_max=tv_iter_max)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gaptv), mean(ssim_gaptv), tgaptv)+'-'*20)
    show_n_save_res(vgaptv,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max = iter_max)

# %%
### [2.1.2] ADMM-TV
if ('all' in test_algo_flag) or ('admmtv' in test_algo_flag):
    projmeth = 'admm' # projection method
    _lambda = 1 # regularization factor, [original set]
    # gamma = 0.01 # parameter in ADMM projection (greater for more noisy data), [original set]
    gamma = 0.05
    denoiser = 'tv' # total variation (TV)
    iter_max = 100 # maximum number of iterations
    # tv_weight = 0.3 # TV denoising weight (larger for smoother but slower) [original set]
    tv_weight = 0.25 
    tv_iter_max = 5 # TV denoising maximum number of iterations each

    vadmmtv,tadmmtv,psnr_admmtv,ssim_admmtv,psnrall_admmtv = admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, gamma=gamma,
                                            denoiser=denoiser, iter_max=iter_max, 
                                            tv_weight=tv_weight, 
                                            tv_iter_max=tv_iter_max)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_admmtv), mean(ssim_admmtv), tadmmtv)+'-'*20)
    show_n_save_res(vadmmtv,tadmmtv,psnr_admmtv,ssim_admmtv,psnrall_admmtv, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max = iter_max, gamma=gamma)

# %%
## [2.2] GAP/ADMM-FFDNet
### [2.2.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
if ('all' in test_algo_flag) or ('gapffdnet' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    # _lambda = 1.5
    accelerate = True # enable accelerated version of GAP
    denoiser = 'ffdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
    iter_max = [10, 10, 10, 10] # maximum number of iterations
    # sigma    = [12/255, 6/255] # pre-set noise standard deviation
    # iter_max = [10,10] # maximum number of iterations
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

    vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate,
                                            denoiser=denoiser, model=model, 
                                            iter_max=iter_max, sigma=sigma)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gapffdnet), mean(ssim_gapffdnet), tgapffdnet)+'-'*20)
    show_n_save_res(vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        iter_max = iter_max, sigma=sigma)

### [2.2.2] ADMM-FFDNet (FFDNet-based frame-wise video denoising)
if ('all' in test_algo_flag) or ('admmffdnet' in test_algo_flag):
    projmeth = 'admm' # projection method
    _lambda = 1 # regularization factor, [original set]
    gamma = 0.05
    denoiser = 'ffdnet' # video non-local network 
    sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
    iter_max = [10, 10, 10, 10] # maximum number of iterations
    # sigma    = [12/255, 6/255] # pre-set noise standard deviation
    # iter_max = [10,10] # maximum number of iterations
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

    vadmmffdnet,tadmmffdnet,psnr_admmffdnet,ssim_admmffdnet,psnrall_admmffdnet = admmdenoise_cacti(meas, mask, A, At,
                                              projmeth=projmeth, v0=None, orig=orig,
                                              iframe=iframe, nframe=nframe,
                                              MAXB=MAXB, maskdirection='plain',
                                              _lambda=_lambda, gamma=gamma,
                                              denoiser=denoiser, iter_max=iter_max, model=model, 
                                              sigma=sigma)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_admmffdnet), mean(ssim_admmffdnet), tadmmffdnet)+'-'*20)
    show_n_save_res(vadmmffdnet,tadmmffdnet,psnr_admmffdnet,ssim_admmffdnet,psnrall_admmffdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        iter_max = iter_max, sigma=sigma, gamma=gamma)

# %%
## [2.3] GAP/ADMM-FastDVDnet
### [2.3.1] GAP-FastDVDnet
if ('all' in test_algo_flag) or ('gapfastdvdnet' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    # _lambda = 1.5
    accelerate = True # enable accelerated version of GAP
    denoiser = 'fastdvdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    sigma    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation
    iter_max = [20, 20, 20, 20] # maximum number of iterations
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True # use GPU

    # pre-load the model for fastdvdnet image denoising
    NUM_IN_FR_EXT = 5 # temporal size of patch
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)

    # Load saved weights
    state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
    if useGPU:
        device_ids = [0]
        # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        model = model.cuda()
    # else:
        # # CPU mode: remove the DataParallel wrapper
        # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        
    model.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate, 
                                            denoiser=denoiser, model=model, 
                                            iter_max=iter_max, sigma=sigma)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gapfastdvdnet), mean(ssim_gapfastdvdnet), tgapfastdvdnet)+'-'*20)
    show_n_save_res(vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        iter_max = iter_max, sigma=sigma)
    
### [2.3.2] ADMM-FastDVDnet
if ('all' in test_algo_flag) or ('admmfastdvdnet' in test_algo_flag):
    projmeth = 'admm' # projection method
    _lambda = 1 # regularization factor, [original set]
    gamma = 0.05
    denoiser = 'fastdvdnet' # video non-local network 
    sigma    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation
    iter_max = [20, 20, 20, 20] # maximum number of iterations
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True # use GPU

    # pre-load the model for fastdvdnet image denoising
    NUM_IN_FR_EXT = 5 # temporal size of patch
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)

    # Load saved weights
    state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
    if useGPU:
        device_ids = [0]
        # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        model = model.cuda()
    # else:
        # # CPU mode: remove the DataParallel wrapper
        # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        
    model.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    vadmmfastdvdnet,tadmmfastdvdnet,psnr_admmfastdvdnet,ssim_admmfastdvdnet,psnrall_admmfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, gamma=gamma,
                                            denoiser=denoiser, model=model, 
                                            iter_max=iter_max, sigma=sigma)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_admmfastdvdnet), mean(ssim_admmfastdvdnet), tadmmfastdvdnet)+'-'*20)
    show_n_save_res(vadmmfastdvdnet,tadmmfastdvdnet,psnr_admmfastdvdnet,ssim_admmfastdvdnet,psnrall_admmfastdvdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        iter_max = iter_max,sigma=sigma, gamma=gamma)

# %%
## [2.4] GAP/ADMM-gaptv+ffdnet
### [2.4.1] GAP-TV+FFDNET
if ('all' in test_algo_flag) or ('gaptv+ffdnet' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv+ffdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    sigma1    = [0] # pre-set noise standard deviation for 1st period denoise 
    iter_max1 = 100 # maximum number of iterations for 1st period denoise   
    sigma2    = [50/255, 20/255, 10/255, 6/255] # pre-set noise standard deviation for 2nd period denoise 
    iter_max2 = [20, 40, 100, 50] # maximum number of iterations for 2nd period denoise    
    # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
    # iter_max2 = [20, 20] # maximum number of iterations for 2nd period denoise   
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    tv_weight = 0.25 # TV denoising weight (larger for smoother but slower)
    tvm = 'tv_chambolle'
    # sigma    = [12/255, 6/255] # pre-set noise standard deviation
    # iter_max = [10,10] # maximum number of iterations
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
    
    vgaptvffdnet,tgaptvffdnet,psnr_gaptvffdnet,ssim_gaptvffdnet,psnrall_gaptvffdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate,
                                            denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                            tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                            model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)
                                            

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gaptvffdnet), mean(ssim_gaptvffdnet), tgaptvffdnet)+'-'*20)
    show_n_save_res(vgaptvffdnet,tgaptvffdnet,psnr_gaptvffdnet,ssim_gaptvffdnet,psnrall_gaptvffdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
  
### [2.4.2] ADMM-TV+FFDNET
if ('all' in test_algo_flag) or ('admmtv+ffdnet' in test_algo_flag):
    projmeth = 'admm' # projection method
    _lambda = 1 # regularization factor, [original set]
    gamma = 0.05
    tvm = 'tv_chambolle'
    # accelerate = True # enable accelerated version of GAP
    denoiser = 'tv+ffdnet' # video non-local network 
    sigma1    = [0] # pre-set noise standard deviation for 1st period denoise 
    iter_max1 = 25 # maximum number of iterations for 1st period denoise   
    sigma2    = [50/255, 20/255, 10/255] # pre-set noise standard deviation for 2nd period denoise 
    iter_max2 = [40, 20, 10] # maximum number of iterations for 2nd period denoise    
    # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
    # iter_max2 = [20, 20] # maximum number of iterations for 2nd period denoise   
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    tv_weight = 0.25 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
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

    vadmmtvffdnet,tadmmtvffdnet,psnr_admmtvffdnet,ssim_admmtvffdnet,psnrall_admmtvffdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, gamma=gamma,
                                            denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                            tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                            model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_admmtvffdnet), mean(ssim_admmtvffdnet), tadmmtvffdnet)+'-'*20)
    show_n_save_res(vadmmtvffdnet,tadmmtvffdnet,psnr_admmtvffdnet,ssim_admmtvffdnet,psnrall_admmtvffdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2, gamma=gamma)

# %%
## [2.5] GAP/ADMM-gaptv+fastdvdnet
import torch
from packages.fastdvdnet.models import FastDVDnet

### [2.5.1] GAP-TV+FASTDVDNET
if ('all' in test_algo_flag) or ('gaptv+fastdvdnet' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv+fastdvdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    sigma1    = [0] # pre-set noise standard deviation for 1st period denoise 
    iter_max1 = 40 # maximum number of iterations for 1st period denoise   
    sigma2    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation for 2nd period denoise 
    iter_max2 = [20, 20, 20, 20] # maximum number of iterations for 2nd period denoise    
    # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
    # iter_max2 = [20, 20] # maximum number of iterations for 2nd period denoise   
    tvm = 'tv_chambolle'
    # tvm = 'tv_bregman'
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    tv_weight = 0.5 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True # use GPU

    # pre-load the model for fastdvdnet image denoising
    NUM_IN_FR_EXT = 5 # temporal size of patch
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)

    # Load saved weights
    state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
    if useGPU:
        device_ids = [0]
        # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        model = model.cuda()
    # else:
        # # CPU mode: remove the DataParallel wrapper
        # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        
    model.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    vgaptvfastdvdnet,tgaptvfastdvdnet,psnr_gaptvfastdvdnet,ssim_gaptvfastdvdnet,psnrall_gaptvfastdvdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate,
                                            denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                            tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                            model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gaptvfastdvdnet), mean(ssim_gaptvfastdvdnet), tgaptvfastdvdnet)+'-'*20)
    show_n_save_res(vgaptvfastdvdnet,tgaptvfastdvdnet,psnr_gaptvfastdvdnet,ssim_gaptvfastdvdnet,psnrall_gaptvfastdvdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
 
 ### [2.5.2] ADMM-TV+FASTDVDNET

### [2.5.2] ADMM-TV+FASTDVDNET
if ('all' in test_algo_flag) or ('admmtv+fastdvdnet' in test_algo_flag):
    projmeth = 'admm' # projection method
    _lambda = 1 # regularization factor, [original set]
    gamma = 0.05
    # accelerate = True # enable accelerated version of GAP
    denoiser = 'tv+fastdvdnet' # video non-local network 
    sigma1    = [0] # pre-set noise standard deviation for 1st period denoise 
    iter_max1 = 40 # maximum number of iterations for 1st period denoise   
    sigma2    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation for 2nd period denoise 
    iter_max2 = [20, 20, 20, 20] # maximum number of iterations for 2nd period denoise    
    # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
    tvm = 'tv_chambolle'
    # tvm = 'tv_bregman'
    # iter_max2 = [20, 20] # maximum number of iterations for 2nd period denoise   
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    tv_weight = 0.5 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True # use GPU

    # pre-load the model for fastdvdnet image denoising
    NUM_IN_FR_EXT = 5 # temporal size of patch
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)

    # Load saved weights
    state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
    if useGPU:
        device_ids = [0]
        # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        model = model.cuda()
    # else:
        # # CPU mode: remove the DataParallel wrapper
        # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        
    model.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    vadmmtvfastdvdnet,tadmmtvfastdvdnet,psnr_admmtvfastdvdnet,ssim_admmtvfastdvdnet,psnrall_admmtvfastdvdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, gamma=gamma,
                                            denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                            tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                            model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_admmtvfastdvdnet), mean(ssim_admmtvfastdvdnet), tadmmtvfastdvdnet)+'-'*20)
    show_n_save_res(vadmmtvfastdvdnet,tadmmtvfastdvdnet,psnr_admmtvfastdvdnet,ssim_admmtvfastdvdnet,psnrall_admmtvfastdvdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+datname, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2, gamma=gamma)
         
# show res
# if show_res_flag:
#     plt.show()

# save params
if save_param_flag:
    # params path
    param_dir = resultsdir+'/savedfig/'
    param_name = 'param_finetune.txt'
    
    if not os.path.exists(param_dir):
            os.makedirs(param_dir) 
            
    param_path = param_dir + param_name
    if os.path.exists(param_path):
        writemode = 'a+'
    else:
        writemode = 'w'
    
    with open(param_path, writemode) as f:
        # customized contents
        f.write(projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M')+':\n')
        f.write('\titer_max1 = ' + str(iter_max1) + '\n')
        f.write('\tsigma2 = ' + str([255*x for x in sigma2]) + '/255\n')
        f.write('\titer_max2 = ' + str(iter_max2) + '\n')
        f.write('\ttv_iter_max = ' + str(tv_iter_max) + '\n')
        f.write('\ttv_weight = ' + str(tv_weight) + '\n')