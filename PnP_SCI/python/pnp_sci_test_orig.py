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
import math
import h5py
import numpy as np
import scipy.io as sio
from statistics import mean
from pnp_sci_algo import admmdenoise_cacti
from joint_pnp_sci_algo import joint_admmdenoise_cacti
from utils import (A_, At_, show_n_save_res)
import matplotlib.pyplot as plt
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version
# import deep models
import torch
from packages.ffdnet.models import FFDNet
from packages.fastdvdnet.models import FastDVDnet



# %%
# [0] environment configuration
# GPU assign
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## [0.1] path and data name
orig_dir = './dataset/benchmark/orig/bm_256_10f'
mask_dir = './dataset/benchmark/mask' # mask dataset
# orig_dir = r'E:\project\CACTI\experiment\simulation\dataset\simu_data\gray\orig'
# mask_dir = r'E:\project\CACTI\experiment\simulation\dataset\simu_data\gray\mask' # mask dataset


resultsdir = './results' # results dir

# orig_name = 'aerial'                # name of 'orig'
# orig_name = 'crash'
# orig_name = 'drop'
# orig_name = 'kobe'            
# orig_name = 'runner'
orig_name = 'traffic'    
# orig_name = 'football_1024' 

mask_name = 'binary_mask_256_10f'    # name of 'mask'
# mask_name = 'multiplex_shift_binary_mask_1024_10f'    # name of 'mask'
# mask_name = 'combine_binary_mask_256_10f_2_uniform'
# mask_name = 'combine_shift_binary_mask_256_10f'


origpath = orig_dir + '/' + orig_name + '.mat' # path of the .mat orig file
maskpath = mask_dir + '/' + mask_name + '.mat' # path of the .mat mask file


## [0.2] flags and params
show_res_flag = 1           # show results
save_res_flag = 0          # save results
save_param_flag =  0        # save params
# choose algorithms: 
# 'all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet', 
# 'gapfastdvdnet', 'admmfastdvdnet', 'gaptv+ffdnet', 'gaptv+fastdvdnet'
# test_algo_flag = ['all']
# test_algo_flag = ['gaptv', 'admmtv']
test_algo_flag = ['admmffdnet']
# test_algo_flag = ['gapffdnet','admmffdnet']
# test_algo_flag = ['gapfastdvdnet','admmfastdvdnet']
# test_algo_flag = ['admmtv','admmffdnet','admmfastdvdnet']
# test_algo_flag = ['gaptv+ffdnet']		
# test_algo_flag = ['admmtv+ffdnet']		
# test_algo_flag = ['admmtv+fastdvdnet']	
# test_algo_flag = ['gaptv+fastdvdnet']	
# test_algo_flag = ['gaptv', 'gaptv+ffdnet']
# test_algo_flag = ['gaptv', 'gaptv+fastdvdnet']	
# test_algo_flag = ['gaptv+ffdnet', 'gaptv+fastdvdnet']	
# test_algo_flag = ['gaptv', 'gaptv+ffdnet', 'gaptv+fastdvdnet']	
# test_algo_flag = ['admmffdnet','admmtv+ffdnet', 'admmfastdvdnet', 'admmtv+fastdvdnet']	

# noise
gaussian_noise_level = 5
poisson_noise = False


# %%
# [1] load data
if get_matfile_version(_open_file(maskpath, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    origfile = sio.loadmat(origpath) # for '-v7.2' and below .mat file (MATLAB)
    maskfile = sio.loadmat(maskpath)
    orig = np.array(origfile['orig'])
    mask = np.array(maskfile['mask'])
    mask = np.float32(mask)
    orig = np.float32(orig)
else: # MATLAB .mat v7.3
    with h5py.File(origpath, 'r') as origfile: # for '-v7.3' .mat file (MATLAB)
        orig = np.array(origfile['orig'])
        orig = np.float32(orig).transpose((2,1,0))
        
    with h5py.File(maskpath, 'r') as maskfile: # for '-v7.3' .mat file (MATLAB)
        # print(list(file.keys()))
        mask = np.array(maskfile['mask'])
        mask = np.float32(mask).transpose((2,1,0))

#  calc meas
nmask = mask.shape[2]
norig = orig.shape[2]
meas = np.zeros([orig.shape[0], orig.shape[1], norig//nmask])
for i in range(norig//nmask):
    tmp_orig = orig[:,:,i*nmask:(i+1)*nmask]
    meas[:,:,i] = np.sum(tmp_orig*mask, 2)

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
MAXB = 255.

# common parameters and pre-calculation for PnP
# define forward model and its transpose
A  = lambda x :  A_(x, mask) # forward model function handle
At = lambda y : At_(y, mask) # transpose of forward model


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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
    iter_max = 80 # maximum number of iterations
    # tv_weight = 0.3 # TV denoising weight (larger for smoother but slower) [original set]
    tv_weight = 0.5 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
  
### [2.4.2] ADMM-TV+FFDNET
if ('all' in test_algo_flag) or ('admmtv+ffdnet' in test_algo_flag):
    projmeth = 'admm' # projection method
    _lambda = 1 # regularization factor, [original set]
    gamma = 0.0
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
    iter_max1 = 100 # maximum number of iterations for 1st period denoise   
    sigma2    = [100/255, 50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
    iter_max2 = [60, 100, 150] # maximum number of iterations for 2nd period denoise    
    # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
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
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2, gamma=gamma)
         
# show res
# if show_res_flag:
#     plt.show()

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