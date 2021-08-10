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
from os.path import join as opj
import time
import math
import h5py
import numpy as np
import scipy.io as sio
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version
import matplotlib.pyplot as plt
from statistics import mean
from datetime import datetime
import time

from pnp_sci_algo import admmdenoise_cacti
from joint_pnp_sci_algo import joint_admmdenoise_cacti

from utils import (A_, At_, show_n_save_res)
import torch
from packages.ffdnet.models import FFDNet
from packages.fastdvdnet.models import FastDVDnet

# %%
# [0] environment configuration

## flags and params
save_res_flag = 1           # save results
show_res_flag = 1           # show results
save_param_flag =  1        # save params
# test_algo_flag = ['all']		# choose algorithms: 'all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet', 'gapfastdvdnet', 'admmfastdvdnet','gaptv+fastdvdnet'
# test_algo_flag = ['gaptv']
# test_algo_flag = ['admmtv']
test_algo_flag = ['gaptv+fastdvdnet']


# datasetdir = r'E:\project\CACTI\SCI algorithm\[dataset]\#benchmark\simu_data\data\binary_mask_256_10f\bm_256_10f' # dataset
# datasetdir = r'E:\project\CACTI\experiment\real_data\data_cacti\#mask_bin2_20210127\scene' # dataset
# datasetdir = r'E:\project\CACTI\experiment\real_data\data_cacti\#_mask_bin2_20210226\scene' # dataset
# datasetdir = r'E:\project\CACTI\experiment\real_data\data_cacti\#mask_bin2_20210227_lcos2\scene'
datasetdir = r'E:\project\CACTI\experiment\real_data\data\##mask_side_circ_Cr10_20210315\scene'
# datasetdir = r'E:\project\CACTI\experiment\real_data\data\#mask_side_circ_20210312\scene'
resultsdir = './results/20210315_blk' # results

# # datname = 'data_kobe'        # name of the dataset
# datname = 'data_d_hand_1_20210127_roi690-180_sz1024'
# datname = 'data_d_alpha2_1_20210127_roi640-350_sz1024'
# datname = 'data_box21_20210226_roi550-240_sz1024'
# datname = 'data_fan16_20210227_roi820-830_sz256'
datname = 'data_opto_985_20210315_roi1975-560_sz400_blk'

# resultsdir = opj('./results',datname) # results
matfile = datasetdir + '/' + datname + '.mat' # path of the .mat data file

# block params

# %%
# [1] load data
if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
    order = 'K' # [order] keep as the default order in Python/numpy
    meas_blk = np.float32(file['meas'], order=order)
    mask_blk = np.float32(file['mask'], order=order)
    # orig = np.float32(file['orig'], order=order)
else: # MATLAB .mat v7.3
    file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    order = 'F' # [order] switch to MATLAB array order
    meas_blk = np.float32(file['meas'], order=order).transpose()
    mask_blk = np.float32(file['mask'], order=order).transpose()
    # orig = np.float32(file['orig'], order=order).transpose()
orig = None # no orig - ground truthcrop_
print('shape of meas_blk & masks_blk',meas_blk.shape, mask_blk.shape)

# judge data saving type (block or normal)
if mask_blk.ndim!=4:
    print("WARNING: Not block type data, please use other codes")
else:
    print("\n===> Starting processing block data...")
    
    
    time_start=time.time()
    for k in range(mask_blk.shape[3]):
        mask = np.squeeze(mask_blk[:,:,:,k])
        
        if meas_blk.ndim == 3:
            meas = np.squeeze(meas_blk[:,:,k])
            recon = np.zeros_like(mask_blk)
        elif meas_blk.ndim == 4:
            mask = np.squeeze(mask_blk[:,:,:,k])
            recon = np.zeros((mask_blk.shape[0],mask_blk.shape[1], mask_blk.shape[2]*meas_blk.shape[2],mask_blk.shape[3]))
        # print(recon.shape)
        
        mask_sum = np.sum(mask, axis=2)
        mask_sum[mask_sum==0] = 1

        # zzh: expand dim for a single 'meas'
        if meas.ndim<3:
            meas = np.expand_dims(meas,2)
            # print(meas.shape)
        # print('meas, mask, orig:', meas.shape, mask.shape, orig.shape)
        
        # normalize data
        mask_max = np.max(mask) 
        mask = mask/mask_max
        meas = meas/mask_max         

        # --------- param ------------
        iframe = 0
        nframe = 1
        nmask = mask.shape[2]

        # MAXB = 255. # for 8bit
        # MAXB = 65535. # for 16bit
        MAXB = 65535/nmask # real measurement's data range is Cr times less then simulated measment
        # print(MAXB)
        # --------- param ------------

        # nframe = meas.shape[2]


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
            iter_max = 20 # maximum number of iterations
            # tv_weight = 0.25 # TV denoising weight (larger for smoother but slower) [kobe:0.25; ]
            tv_weight = 1 # TV denoising weight
            tv_iter_max = 5 # TV denoising maximum number of iterations each
            vdenoise,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,
                                                    denoiser=denoiser, iter_max=iter_max, 
                                                    tv_weight=tv_weight, 
                                                    tv_iter_max=tv_iter_max)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(),  tgaptv)+'-'*20)
            # show_n_save_res(vgaptv,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     tv_weight=tv_weight, iter_max = iter_max)

        # %%
        ### [2.1.2] ADMM-TV
        if ('all' in test_algo_flag) or ('admmtv' in test_algo_flag):
            projmeth = 'admm' # projection method
            _lambda = 1 # regularization factor, [original set]
            # gamma = 0.01 # parameter in ADMM projection (greater for more noisy data), [original set]
            # gamma = 0.05
            gamma = 0.2
            denoiser = 'tv' # total variation (TV)
            iter_max = 10 # maximum number of iterations
            # tv_weight = 0.3 # TV denoising weight (larger for smoother but slower) [original set]
            tv_weight = 0.5
            # tv_iter_max = 5 # TV denoising maximum number of iterations each
            tv_iter_max = 5

            vdenoise,tadmmtv,psnr_admmtv,ssim_admmtv,psnrall_admmtv = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, gamma=gamma,
                                                    denoiser=denoiser, iter_max=iter_max, 
                                                    tv_weight=tv_weight, 
                                                    tv_iter_max=tv_iter_max)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tadmmtv)+'-'*20)
            # show_n_save_res(vadmmtv,tadmmtv,psnr_admmtv,ssim_admmtv,psnrall_admmtv, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     tv_weight=tv_weight, iter_max = iter_max, gamma=gamma)
                
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

            vdenoise,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=iter_max, sigma=sigma)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tgapffdnet)+'-'*20)
            # show_n_save_res(vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     iter_max = iter_max, sigma=sigma)

        ### [2.2.2] ADMM-FFDNet (FFDNet-based frame-wise video denoising)
        if ('all' in test_algo_flag) or ('admmffdnet' in test_algo_flag):
            projmeth = 'admm' # projection method
            _lambda = 1 # regularization factor, [original set]
            gamma = 2
            denoiser = 'ffdnet' # video non-local network 
            sigma    = [100/255, 50/255, 25/255, 6/255] # pre-set noise standard deviation
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

            vdenoise,tadmmffdnet,psnr_admmffdnet,ssim_admmffdnet,psnrall_admmffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, gamma=gamma,
                                                    denoiser=denoiser, iter_max=iter_max, model=model, 
                                                    sigma=sigma)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tadmmffdnet)+'-'*20)
            # show_n_save_res(vadmmffdnet,tadmmffdnet,psnr_admmffdnet,ssim_admmffdnet,psnrall_admmffdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     iter_max = iter_max, sigma=sigma, gamma=gamma)

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

            vdenoise,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate, 
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=iter_max, sigma=sigma)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tgapfastdvdnet)+'-'*20)
            # show_n_save_res(vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     iter_max = iter_max, sigma=sigma)
            
        ### [2.3.2] ADMM-FastDVDnet
        if ('all' in test_algo_flag) or ('admmfastdvdnet' in test_algo_flag):
            projmeth = 'admm' # projection method
            _lambda = 1 # regularization factor, [original set]
            # _lambda = 0.7 # regularization factor, [original set]
            gamma = 0.5
            denoiser = 'fastdvdnet' # video non-local network 
            # sigma    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation
            sigma    = [0.5, 0.2, 0.05, 0.01] # pre-set noise standard deviation
            iter_max = [40, 30, 10, 10] # maximum number of iterations
            # sigma    = [12/255] # pre-set noise standard deviation
            # iter_max = [20] # maximum number of iterations
            useGPU = True # use GPU

            # pre-load the model for fastdvdnet image denoising
            # NUM_IN_FR_EXT = 5 # temporal size of patch
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

            vdenoise,tadmmfastdvdnet,psnr_admmfastdvdnet,ssim_admmfastdvdnet,psnrall_admmfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, gamma=gamma,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=iter_max, sigma=sigma)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tadmmfastdvdnet)+'-'*20)
            # show_n_save_res(vadmmfastdvdnet,tadmmfastdvdnet,psnr_admmfastdvdnet,ssim_admmfastdvdnet,psnrall_admmfastdvdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     iter_max = iter_max,sigma=sigma, gamma=gamma)

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
            
            vdenoise,tgaptvffdnet,psnr_gaptvffdnet,ssim_gaptvffdnet,psnrall_gaptvffdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,
                                                    denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                                    tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                                    model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)
                                                    

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tgaptvffdnet)+'-'*20)
            # show_n_save_res(vgaptvffdnet,tgaptvffdnet,psnr_gaptvffdnet,ssim_gaptvffdnet,psnrall_gaptvffdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
        
        ### [2.4.2] ADMM-TV+FFDNET
        if ('all' in test_algo_flag) or ('admmtv+ffdnet' in test_algo_flag):
            projmeth = 'admm' # projection method
            _lambda = 1 # regularization factor, [original set]
            # gamma = 0.05 # [original set]
            gamma = 4
            tvm = 'tv_chambolle'
            # accelerate = True # enable accelerated version of GAP
            denoiser = 'tv+ffdnet' # video non-local network 
            noise_estimate = False # disable noise estimation for GAP
            sigma1    = [0] # pre-set noise standard deviation for 1st period denoise 
            iter_max1 = 60 # maximum number of iterations for 1st period denoise  
            sigma2    = [100/255, 50/255, 20/255, 10/255] # pre-set noise standard deviation for 2nd period denoise , [original set]
            iter_max2 = [20, 20, 10, 10] # maximum number of iterations for 2nd period denoise   
            # sigma2    = [50/255, 20/255, 10/255, 6/255] # pre-set noise standard deviation for 2nd period denoise , [original set]
            # iter_max2 = [10, 10, 10, 10] # maximum number of iterations for 2nd period denoise    
            # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
            # iter_max2 = [20, 20] # maximum number of iterations for 2nd period denoise   
            tv_iter_max = 5 # TV denoising maximum number of iterations each
            # tv_weight = 0.25 # TV denoising weight (larger for smoother but slower) [kobe:0.25], [original set]
            tv_weight = 2 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
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

            vdenoise,tadmmtvffdnet,psnr_admmtvffdnet,ssim_admmtvffdnet,psnrall_admmtvffdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, gamma=gamma,
                                                    denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                                    tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                                    model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tadmmtvffdnet)+'-'*20)
            # show_n_save_res(vadmmtvffdnet,tadmmtvffdnet,psnr_admmtvffdnet,ssim_admmtvffdnet,psnrall_admmtvffdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2, gamma=gamma)

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
            iter_max1 = 50 # maximum number of iterations for 1st period denoise   
            sigma2    = [150/MAXB, 80/MAXB, 50/MAXB, 30/MAXB] # pre-set noise standard deviation for 2nd period denoise 
            iter_max2 = [60, 60, 60, 60] # maximum number of iterations for 2nd period denoise                 
            tv_iter_max = 5 # TV denoising maximum number of iterations each
            tv_weight = 12 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
            # tv_weight = 0.5 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
            tvm = 'tv_chambolle'
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

            vdenoise,tgaptvfastdvdnet,psnr_gaptvfastdvdnet,ssim_gaptvfastdvdnet,psnrall_gaptvfastdvdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,
                                                    denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                                    tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                                    model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tgaptvfastdvdnet)+'-'*20)
            # show_n_save_res(vgaptvfastdvdnet,tgaptvfastdvdnet,psnr_gaptvfastdvdnet,ssim_gaptvfastdvdnet,psnrall_gaptvfastdvdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag, tv_iter_max=tv_iter_max,
            #                     tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
        
        ### [2.5.2] ADMM-TV+FASTDVDNET
        if ('all' in test_algo_flag) or ('admmtv+fastdvdnet' in test_algo_flag):
            projmeth = 'admm' # projection method
            _lambda = 1 # regularization factor, [original set]
            gamma = 0.05 # [original set]
            # gamma = 0.1
            # accelerate = True # enable accelerated version of GAP
            denoiser = 'tv+fastdvdnet' # video non-local network 
            sigma1    = [0] # pre-set noise standard deviation for 1st period denoise 
            iter_max1 = 40 # maximum number of iterations for 1st period denoise   
            sigma2    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation for 2nd period denoise 
            iter_max2 = [20, 20, 20, 20] # maximum number of iterations for 2nd period denoise    
            # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
            # iter_max2 = [20, 20] # maximum number of iterations for 2nd period denoise   
            tv_iter_max = 5 # TV denoising maximum number of iterations each
            tvm = 'tv_chambolle'
            # tv_weight = 1 # 
            tv_weight = 0.5 # TV denoising weight (larger for smoother but slower) [kobe:0.25] [original set]
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

            vdenoise,tadmmtvfastdvdnet,psnr_admmtvfastdvdnet,ssim_admmtvfastdvdnet,psnrall_admmtvfastdvdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, gamma=gamma,
                                                    denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                                    tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                                    model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

            print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
                projmeth.upper(), denoiser.upper(), tadmmtvfastdvdnet)+'-'*20)
            # show_n_save_res(vadmmtvfastdvdnet,tadmmtvfastdvdnet,psnr_admmtvfastdvdnet,ssim_admmtvfastdvdnet,psnrall_admmtvfastdvdnet, orig, nmask, resultsdir, 
            #                     projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
            #                     show_res_flag=show_res_flag, save_res_flag=save_res_flag,
            #                     tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2, gamma=gamma)
        # save recon
        if meas_blk.ndim == 3:
            recon[:,:,:,k] = vdenoise
            
        time_end=time.time()
        print("===> Finished block {} / {},   Total time {:.2f} min".format(k+1, mask_blk.shape[3], (time_end-time_start)/60))


# save res
show_n_save_res(recon, time_end/60, 0, 0, 0, orig, nmask, resultsdir, 
                projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe,nframe=nframe, MAXB=MAXB, 
                show_res_flag=0, save_res_flag=save_res_flag, tv_iter_max=tv_iter_max,
                tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
                   
# show res
if show_res_flag:
    plt.show()

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