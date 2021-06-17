''' Utilities '''
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import scipy.io as sio
import os
import cv2

def A_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return np.sum(x*Phi, axis=2)  # element-wise product

def At_(y, Phi):
    '''
    Tanspose of the forward model. 
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = np.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



## zzh
def show_n_save_res(vdenoise,tdenoise,psnr_denoise,ssim_denoise,psnrall_denoise, orig, Cr, resultsdir, 
                    save_name, iframe=0, nframe=1, MAXB=255, show_res_flag=1, save_res_flag=1, **kwargs):
    # show res
    if show_res_flag:
        # setting
        col_num = Cr//2
        fig_sz = (12, 6.5)
        savedfigdir = resultsdir + '/savedfig/'
        if not os.path.exists(savedfigdir):
            os.makedirs(savedfigdir)      
        
        # fig
        for kf in range(nframe):
            if orig is not None: #  ground truth is valid
                orig_k = orig[:,:,(kf+iframe)*Cr:(kf+iframe+1)*Cr]/MAXB  
                # plt.ion() # interactive mode
                fig = plt.figure(figsize=fig_sz)
                for nt in range(Cr):
                    plt.subplot(Cr//col_num, col_num, nt+1)
                    plt.imshow(orig_k[:,:,nt], cmap=plt.cm.gray, vmin=0, vmax=1)
                    plt.axis('off')
                    plt.title('Ground truth: Frame #{0:d}'.format((kf+iframe)*Cr+nt+1), fontsize=12)
                plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0, top=1, left=0, right=1)
                plt.savefig('{}{}_kmeas{:d}_orig.png'.format(savedfigdir,save_name,kf+iframe)) 

            vdenoise_k = vdenoise[:,:,kf*Cr:(kf+1)*Cr]
            fig = plt.figure(figsize=fig_sz)
            for nt in range(Cr):
                plt.subplot(Cr//col_num, col_num, nt+1)
                plt.imshow(vdenoise_k[:,:,nt], cmap=plt.cm.gray, vmin=0, vmax=1)
                plt.axis('off')
                if orig is not None:
                    plt.title('Frame #{0:d} ({1:2.2f} dB)'.format((kf+iframe)*Cr+nt+1,psnr_denoise[nt]), fontsize=12)
                else:
                    plt.title('Frame #{0:d})'.format((kf+iframe)*Cr+nt+1), fontsize=12)
            # PSNR_rec = np.zeros(Cr)    
            # print('Mean PSNR {:2.2f} dB.'.format(mean(psnr_denoise)))
            # plt.title('-{} mean PSNR {:2.2f} dB'.format(denoiser.upper(),np.mean(PSNR_rec)))
            plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0, top=1, left=0, right=1)
            plt.savefig('{}{}_kmeas{:d}_vdenoise.png'.format(savedfigdir,save_name,kf+iframe)) 

            if orig is not None:
                plt.figure()
                plt.plot(psnrall_denoise[kf], 'r')
                plt.savefig('{}{}_kmeas{:d}_psnr_all.png'.format(savedfigdir,save_name,kf+iframe)) 
        if orig is not None:
            plt.figure()
            # plt.rcParams["font.family"] = 'monospace'
            # plt.rcParams["font.size"] = "20"
            plt.plot(psnr_denoise)
            # plt.plot(psnr_denoise,color='black')
            plt.savefig('{}{}_psnr_framewise.png'.format(savedfigdir,save_name)) 


        # plt.ioff()
        
    # save res
    if save_res_flag:
        savedmatdir = resultsdir + '/savedmat/'
        if not os.path.exists(savedmatdir):
            os.makedirs(savedmatdir)
        print('Results saved to: {}{}_kmeas{:d}_{:d}.mat\n'.format(savedmatdir,save_name,iframe,iframe+nframe-1))
        if orig is not None:
            sio.savemat('{}{}_kmeas{:d}_{:d}.mat'.format(savedmatdir,save_name,iframe,iframe+nframe-1),
                        {'vdenoise':vdenoise,
                         'orig':orig, 
                        'psnr_denoise':psnr_denoise,
                        'ssim_denoise':ssim_denoise,
                        'psnrall_denoise':psnrall_denoise,
                        'psnr_mean':mean(psnr_denoise),
                        'tdenoise':tdenoise,
                        'iframe':iframe,
                        'nframe':nframe,
                        'Cr':Cr,
                        **kwargs                 
                        })
        else:
            sio.savemat('{}{}_kmeas{:d}_{:d}.mat'.format(savedmatdir,save_name,iframe,iframe+nframe-1),
                        {'vdenoise':vdenoise, 
                        'tdenoise':tdenoise,
                        'iframe':iframe,
                        'nframe':nframe,
                        'Cr':Cr,
                        **kwargs                 
                        })

# save results to images
def save_rgb_img(img, save_dir, prefix='img', save_format='.jpg', rescale_ch=False):
    img_num = img.shape[-1]
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if rescale_ch:
        # rescale each channel separately
        for ch in range(3):
            img[:,:,ch,:] = rescale(img[:,:,ch,:])
            
    for k in range(img_num):
        tmp_img = np.uint8(img[:,:,:,k]*255)
        tmp_img = tmp_img[...,::-1] # RGB 2 BGR for cv2
        save_path = os.path.join(save_dir,prefix+'%04d'%k+save_format)
        cv2.imwrite(save_path, tmp_img)
        
    print('images saved to: ', save_dir)


# CLI script
def cli_run(script_name, orig_name, scale, Cr, mask_name, test_algo_flag,
            root_dir = '.', result_path='/results/tmp', iframe=0, nframe=1, MAXB=255,
            show_res_flag=0, save_res_flag=0, log_result_flag=0,
            gaussian_noise_level=0, poisson_noise=0, gamma=0,
            tv_weight=None, iter_max1=0, sigma1=0, iter_max2=[0], sigma2=[0]):

    command_str = ('python {} \
        --orig_name {} \
        --scale {} \
        --Cr {} \
        --mask_name {} \
        --test_algo_flag {} \
        --root_dir {} \
        --result_path {} \
        --iframe {} \
        --nframe {} \
        --MAXB {} \
        --show_res_flag {} \
        --save_res_flag {} \
        --log_result_flag {} \
        --gaussian_noise_level {} \
        --poisson_noise {} \
        --gamma {} \
        --tv_weight {:.2f} \
        --iter_max1 {} \
        --sigma1 {} ' +
        '--iter_max2 ' + (' {} '*len(iter_max2)) +
        '--sigma2 ' + (' {:.4f} '*len(sigma2))).format(script_name, orig_name, scale, Cr, mask_name, test_algo_flag, root_dir, result_path, iframe, nframe, MAXB,
                show_res_flag, save_res_flag, log_result_flag, gaussian_noise_level, poisson_noise, gamma, tv_weight, iter_max1, sigma1, *iter_max2, *sigma2)
        
    print(command_str)
    os.system(command_str)
    
def rescale(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range