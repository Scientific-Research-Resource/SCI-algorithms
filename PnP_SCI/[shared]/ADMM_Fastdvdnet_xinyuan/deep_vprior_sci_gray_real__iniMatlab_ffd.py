# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:03:10 2020

@author: Xin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:27:55 2020

@author: Xin
"""


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

from dvp_linear_inv import (gap_denoise_bayer, gap_denoise_gray,
                            gap_denoise, admm_denoise,
                            GAP_TV_rec, ADMM_TV_rec)

from utils import (A_, At_, psnr)


# In[2]:


# [0] environment configuration
# datasetdir = './dataset/cacti' # dataset
datasetdir = 'D:/MuQiao/cacti_201910/data_20191013' # dataset
resultsdir = './results/real_aplp' # results

matlab_ffd_result_dir = 'C:/CACTI_mu_results/Gray_scale_results_201909/Result_Muqiao_20191013'
# datname = 'kobe' # name of the dataset
# datname = 'traffic' # name of the dataset
# datname = 'bus_bayer' # name of the dataset
# datname = 'bus_256_bayer' # name of the dataset
# datname = 'traffic_bayer' # name of the dataset#
for ncount in range(1,5):
    maskfile = datasetdir + '/' + 'mask.mat' 
    mfile = sio.loadmat(maskfile)
    mask_all = np.array(mfile['mask'])
    
    if ncount == 0:
        datname = 'hand'
    elif ncount == 1:
        datname = 'duomino'
    elif ncount == 2:
        datname = 'waterBalloon'
    elif ncount == 3:
        datname = 'pendulumBall'
    elif ncount == 4:
        datname = 'pingpang'
    else:
        datname = 'hand'    

    
    for nmask in range(10, 60, 10):
        matfile = datasetdir + '/meas_' + datname + '_cr_' + np.array2string(np.array(nmask)) + '.mat' # path of the .mat data file
        
        if nmask == 10:
                plot_row = 2
                plot_col = 5
        elif nmask == 20:
                plot_row = 4
                plot_col = 5
        elif nmask == 30:
                plot_row = 5
                plot_col = 6
        elif nmask == 40:
                plot_row = 5
                plot_col = 8
        elif nmask == 50:
                plot_row = 5
                plot_col = 10
        else:
                plow_row = 1
                plot_col = 2
                
        file = sio.loadmat(matfile)
        meas = np.array(file['meas'])


        
        #==============================================================================
        
        mask = np.float32(mask_all[:,:,0:(nmask)])
        meas = np.float32(meas)
        #orig = np.float32(orig)  no orig here
        if len(meas.shape) >= 3:
            (nrows, ncols,nmea) = meas.shape
        else:
            (nrows, ncols) = meas.shape
            nmea = 1
        #(nrows, ncols,nmask) = mask.shape
        
        # now load the matlab results of FFDnet
        ffdresult_filename = matlab_ffd_result_dir + '/meas_' + datname + '_cr_' + np.array2string(np.array(nmask)) + '_FFDnet.mat' 
        with h5py.File(ffdresult_filename, 'r') as ffd_file:
            vgaptv = np.array(ffd_file['im_TV_save'])
            vgaptvffd = np.array(ffd_file['im_TV_save'])
            vffd = np.array(ffd_file['im_ffd_save'])
        
        
        vffd_ini = np.float32(vffd).transpose(2,1,0)
        
        vgap_fastdvd_gray = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
        
           
        for iframe in range(nmea):
            MAXB = 255.
            
            if len(meas.shape) >= 3:
                meas_t = (meas[:,:,iframe])/MAXB*nmask/2
            else:
                meas_t = meas/MAXB*nmask/2
            #orig_t = orig[:,:,iframe*nmask:(iframe+1)*nmask]/MAXB
            
            A  = lambda x :  A_(x, mask) # forward model function handle
            At = lambda y : At_(y, mask) # transpose of forward model
            mask_sum = np.sum(mask, axis=2)
            mask_sum[mask_sum==0] = 1
            
            vffd_ini_t = vffd_ini[:,:,iframe*nmask:((iframe+1)*nmask)]
            '''
            plt.figure
            for np in range(nmask):
                temp_p = (vffd_ini_t[:,:,np])
                plt.subplot(plot_row, plot_col,np+1)
                plt.gray() 
                plt.imshow(temp_p/vffd_ini_t.max())
            plt.show
            plt.pause(1)
            '''
            # In[11]:
            
            import torch
            from packages.fastdvdnet.models import FastDVDnet
            
            ## [2.2] GAP-FastDVDnet
            _lambda = 1 # regularization factor
            accelerate = True # enable accelerated version of GAP
            denoiser = 'fastdvdnet' # video non-local network 
            noise_estimate = False # disable noise estimation for GAP
            # sigma    = [50/255, 25/255, 12/255, 6/255, 3/255] # pre-set noise standard deviation
            # iter_max = [10, 10, 10, 10, 10] # maximum number of iterations
            sigma    = [20/255, 15/255] # pre-set noise standard deviation
            iter_max = [20, 20] # maximum number of iterations
            # sigma    = [50/255,25/255] # pre-set noise standard deviation
            # iter_max = [10,10] # maximum number of iterations
            useGPU = True # use GPU
            
            # pre-load the model for FFDNet image denoising
            NUM_IN_FR_EXT = 5 # temporal size of patch
            model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)
        
            
            # Load saved weights
            state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
            if useGPU:
                device_ids = [0]
                #model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
                model = model.cuda()
            else:
                # CPU mode: remove the DataParallel wrapper
                state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
            model.load_state_dict(state_temp_dict)
            
            # Sets the model in evaluation mode (e.g. it removes BN)
            model.eval()
            
            gamma =0.1 # noise level
            begin_time = time.time()
            vgapfastdvdnet_gray_t=  admm_denoise(meas_t,  mask_sum, A,At, _lambda, gamma,
                                             denoiser, iter_max, 
                                              noise_estimate, sigma,
                                              x0= vffd_ini_t,
                                              X_orig=None, 
                                              model=model)
            
            end_time = time.time()
            tgapfastdvdnet_gray = end_time - begin_time
            print('ADMM-{}  running time {:.1f} seconds.'.format(denoiser.upper(), tgapfastdvdnet_gray))
            
            tem_fastdvd= vgapfastdvdnet_gray_t[0] 
            '''
            plt.figure
            for np in range(nmask):
                temp_p = (tem_fastdvd[:,:,np])
                plt.subplot(plot_row, plot_col,np+1)
                plt.gray() 
                plt.imshow(temp_p/tem_fastdvd.max())
            plt.show
            plt.pause(1)
            '''
            vgap_fastdvd_gray[:,:,iframe*nmask:(iframe+1)*nmask] = tem_fastdvd
          
            # In[12]:
          
            
        savedmatdir = resultsdir + '/savedmat/'
        if not os.path.exists(savedmatdir):
            os.makedirs(savedmatdir)
        # sio.savemat('{}gap{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
        #             {'vgapdenoise':vgapdenoise},{'psnr_gapdenoise':psnr_gapdenoise})
        sio.savemat('{}gap{}_{}{:d}_sigma{:d}_1020b.mat'.format(savedmatdir,denoiser.lower(),datname,nmask,int(sigma[-1]*MAXB)),
                    {'vgap_fastdvd_gray':vgap_fastdvd_gray,
                     'tgapfastdvdnet_gray':tgapfastdvdnet_gray,
                     'meas':meas})


'''
            # In[4]:
            
            
            ## [2.1] GAP-TV [for baseline reference]
            _lambda = 1 # regularization factor
            accelerate = True # enable accelerated version of GAP
            denoiser = 'tv' # total variation (TV)
            iter_max = 100 # maximum number of iterations
            tv_weight = 1 # TV denoising weight (larger for smoother but slower)
            tv_iter_max = 5 # TV denoising maximum number of iterations each
            begin_time = time.time()
            gamma = 0.5

            vgaptv_t= admm_denoise(meas_t, mask_sum, A,At, _lambda, gamma, denoiser, iter_max, 
                                          tv_weight=tv_weight, 
                                          tv_iter_max=tv_iter_max,
                                          X_orig=None,model = None)
            end_time = time.time()
            tgaptv = end_time - begin_time
            tem= vgaptv_t[0] 
            plt.figure
            for np in range(nmask):
                temp_p = (tem[:,:,np])
                plt.subplot(plot_row, plot_col,np+1)
                plt.gray() 
                plt.imshow(temp_p)
            plt.pause(1)

            print('ADMM-{}, running time {:.1f} seconds.'.format(
                denoiser.upper(), tgaptv))
            vgaptv[:,:,iframe*nmask:(iframe+1)*nmask] =  tem
            #psnr_gaptv[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gaptv_t,(nmask,1))
            #ssim_gaptv[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gaptv_t,(nmask,1))
            
            
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
            # iter_max = [20,20,20,10] # maximum number of iterations
            sigma    = [50/255, 30/255, 15/255, 10/255] # pre-set noise standard deviation
            #sigma    = [75, 50, 25, 12, 6] # pre-set noise standard deviation
            iter_max = [30,20,20, 20] # maximum number of iterations
            #sigma    = [50/255,25/255, 12/255, 10/255, 15/255] # pre-set noise standard deviation
            #iter_max = [20, 20, 20, 30,2] # maximu
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
            
            
            gamma =0.1
            
            begin_time = time.time()
            vgapffdnet_t =  admm_denoise(meas_t,  mask_sum, A,At, _lambda, gamma,
                                             denoiser, iter_max, 
                                              noise_estimate, sigma,
                                              x0= None,
                                              X_orig=None, 
                                              model=model)
            
            end_time = time.time()
            tgapffdnet = end_time - begin_time
            tem_ffd= vgapffdnet_t[0] 
            plt.figure
            for np in range(nmask):
                temp_p = (tem_ffd[:,:,np])
                plt.subplot(plot_row, plot_col,np+1)
                plt.gray() 
                plt.imshow(temp_p/tem_ffd.max())
                #plt.imshow(temp_p)
            #plt.show
            plt.pause(1)
            
            print('ADMM-{} running time {:.1f} seconds.'.format( denoiser.upper(),tgapffdnet))
            vgapffdgray[:,:,iframe*nmask:(iframe+1)*nmask] =  tem_ffd
            
            # iniTV
            sigma    = [ 30/255, 15/255, 10/255] # pre-set noise standard deviation
            #sigma    = [75, 50, 25, 12, 6] # pre-set noise standard deviation
            iter_max = [20,20, 20] # maximum number of iterations
            begin_time = time.time()
            vgapffdnet_initv_t =  admm_denoise(meas_t,  mask_sum, A,At, _lambda, gamma,
                                             denoiser, iter_max, 
                                              noise_estimate, sigma,
                                              x0= tem,
                                              X_orig=None, 
                                              model=model)
            
            end_time = time.time()
            tgapffdnet_inttv = end_time - begin_time
            tem_ffd_initv= vgapffdnet_initv_t[0] 
            plt.figure
            for np in range(nmask):
                temp_p = (tem_ffd_initv[:,:,np])
                plt.subplot(plot_row, plot_col,np+1)
                plt.gray() 
                plt.imshow(temp_p/tem_ffd.max())
                #plt.imshow(temp_p)
            #plt.show
            plt.pause(1)
            print('ADMM-{} running time {:.1f} seconds.'.format( denoiser.upper(),tgapffdnet))
            vgapffdgray_initv[:,:,iframe*nmask:(iframe+1)*nmask] =  tem_ffd_initv
'''            