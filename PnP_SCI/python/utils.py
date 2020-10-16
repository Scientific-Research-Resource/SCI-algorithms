''' Utilities '''
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import scipy.io as sio
import os


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
                    save_name, MAXB=255, show_res_flag=1, save_res_flag=1):
    # show res
    if show_res_flag:
        plt.ion()
        fig = plt.figure(figsize=(12, 6.5))
        row_num = 5
        savedfigdir = resultsdir + '/savedfig/cacti/'
        
        for nt in range(Cr):
            plt.subplot(Cr//row_num, row_num, nt+1)
            plt.imshow(orig[:,:,nt]/MAXB, cmap=plt.cm.gray, vmin=0, vmax=1)
            plt.axis('off')
            plt.title('Ground truth: Frame #{0:d}'.format(nt+1), fontsize=12)
        plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0, top=1, left=0, right=1)
        plt.savefig('{}{}_Cr{:d}_orig.png'.format(savedfigdir,save_name,Cr)) 

        fig = plt.figure(figsize=(12, 6.5))
        PSNR_rec = np.zeros(Cr)
        for nt in range(Cr):
            plt.subplot(Cr//row_num, row_num, nt+1)
            plt.imshow(vdenoise[:,:,nt], cmap=plt.cm.gray, vmin=0, vmax=1)
            plt.axis('off')
            plt.title('Frame #{0:d} ({1:2.2f} dB)'.format(nt+1,psnr_denoise[nt]), fontsize=12)
            
        # print('Mean PSNR {:2.2f} dB.'.format(mean(psnr_denoise)))
        # plt.title('-{} mean PSNR {:2.2f} dB'.format(denoiser.upper(),np.mean(PSNR_rec)))
        plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0, top=1, left=0, right=1)
        plt.savefig('{}{}_Cr{:d}_vdenoise.png'.format(savedfigdir,save_name,Cr)) 


        plt.figure()
        # plt.rcParams["font.family"] = 'monospace'
        # plt.rcParams["font.size"] = "20"
        plt.plot(psnr_denoise)
        # plt.plot(psnr_denoise,color='black')
        plt.savefig('{}{}_Cr{:d}_psnr_framewise.png'.format(savedfigdir,save_name,Cr)) 


        plt.figure()
        plt.plot(*psnrall_denoise, 'r')
        plt.savefig('{}{}_Cr{:d}_psnr_all.png'.format(savedfigdir,save_name,Cr)) 

        plt.ioff()
        
    # save res
    if save_res_flag:
        savedmatdir = resultsdir + '/savedmat/cacti/'
        if not os.path.exists(savedmatdir):
            os.makedirs(savedmatdir)
        sio.savemat('{}{}_Cr{:d}.mat'.format(savedmatdir,save_name,Cr),
                    {'vdenoise':vdenoise, 
                    'psnr_denoise':psnr_denoise,
                    'ssim_denoise':ssim_denoise,
                    'psnrall_denoise':psnrall_denoise,
                    'tdenoise':tdenoise
                    })   