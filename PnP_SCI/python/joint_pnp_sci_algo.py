import time
import math
import skimage
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_tv_bregman, 
                                 denoise_bilateral, denoise_wavelet, estimate_sigma)
# from packages.vnlnet.test import vnlnet
from packages.ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser
from packages.fastdvdnet.test_fastdvdnet import fastdvdnet_denoiser
from utils import (A_, At_, psnr)
if skimage.__version__ < '0.18':
    from skimage.measure import (compare_psnr, compare_ssim)
else: # skimage.measure deprecated in version 0.18 ( -> skimage.metrics )
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity   as compare_ssim



# joint admm(gap) denosie cacti (including gap)
def joint_admmdenoise_cacti(meas, mask, A, At, projmeth='admm', v0=None, orig=None, 
                      iframe=0, nframe=1, MAXB=1., maskdirection='plain', denoiser='tv',
                      iter_max1=50, iter_max2=50, sigma1 = None, sigma2=None, **args):
    '''
    Alternating direction method of multipliers (ADMM) or generalized 
    alternating projection (GAP) -based denoising (based on the 
    plug-and-play (PnP) framework) algorithms for video snapshot compressive
    imaging (SCI) or coded aperture compressive temporal imaging (CACTI, 
    Llull et al. Opt. Express 2013).
    '''
    nrow, ncol, nmask = mask.shape
    x_ = np.zeros((nrow,ncol,nmask*nframe), dtype=np.float32)
    psnr_, ssim_, psnrall_ = ([], [], [])
    begin_time = time.time()
    # loop over all the coded frames [nframe]
    for kf in range(nframe):
        print('\n=== %s-%s Reconstruction coded frame block %2d of %2d ==='
              %(projmeth.upper(), denoiser.upper(), kf+1, nframe))
        if orig is not None:
            orig_k = orig[:,:,(kf+iframe)*nmask:(kf+iframe+1)*nmask]/MAXB
        else:
            orig_k = None
        meas_k = meas[:,:,kf+iframe]/MAXB
        if v0 is None:
            v0_k = None
        else: # initialization according to the direction of the masks [up as calibration]
            v0_k = v0[:,:,kf*nmask:(kf+1)*nmask]
            if (maskdirection.lower() == 'updown' and (kf+iframe) % 2 == 1) or \
               (maskdirection.lower() == 'downup' and (kf+iframe) % 2 == 0):  # down (up as mask)
               v0_k = v0_k[:,:,::-1]

        mask_sum = np.sum(mask, axis=2)
        mask_sum[mask_sum==0] = 1
        if projmeth.lower() == 'admm': # alternating direction method of multipliers (ADMM)-based projection
            x_k, psnr_k, ssim_k, psnrall_k = admm_joint_denoise(meas_k, mask_sum, A, At, x0=v0_k, X_orig=orig_k, 
                                                                denoiser=denoiser, iter_max1=iter_max1, 
                                                                iter_max2=iter_max2, sigma1 = sigma1, sigma2=sigma2, **args)

        elif projmeth.lower() == 'gap': # generalized alternating projection (GAP)-based projection
            x_k, psnr_k, ssim_k, psnrall_k =  gap_joint_denoise(meas_k, mask_sum, A, At, x0=v0_k, X_orig=orig_k, 
                                                                denoiser=denoiser, iter_max1=iter_max1, 
                                                                iter_max2=iter_max2, sigma1 = sigma1, sigma2=sigma2, **args)
        else:
            print('Unsupported projection method %s' % projmeth.upper())
        
        if (maskdirection.lower() == 'updown' and (kf+iframe) % 2 == 1) or \
           (maskdirection.lower() == 'downup' and (kf+iframe) % 2 == 0):   # down (up as mask)
            x_k = x_k[:,:,::-1]
            psnr_k = psnr_k[::-1]
            ssim_k = ssim_k[::-1]
            psnrall_k = psnrall_k[::-1]
        
        t_ = time.time() - begin_time
        x_[:,:,kf*nmask:(kf+1)*nmask] = x_k
        psnr_.extend(psnr_k)
        ssim_.extend(ssim_k)
        psnrall_.append(psnrall_k)
        
    return x_, t_, psnr_, ssim_, psnrall_


def admm_joint_denoise(y, Phi_sum, A, At, x0=None, X_orig=None, denoiser='tv+ffdnet', 
                      iter_max1=50, iter_max2=50, sigma1 = None, sigma2=None, **args):
    '''
    ADMM-based joint denoise: multi period and multi step denoise
    
    '''
    # [1] 1st period denoising: gaptv_denoising
    print('*** 1st period denoising ***')
    x, psnr, ssim, psnrall =  admm_denoise(y, Phi_sum, A, At, x0=x0, 
                                           X_orig=X_orig, denoiser='tv', iter_max=iter_max1, sigma=sigma1, **args)  
    
    # [2] 2nd period denoising: gap tv+ffdnet multistep denoising
    print('*** 2nd period denoising ***')
    x, psnr, ssim, psnrall =  admm_multistep_denoise(y, Phi_sum, A, At, 
                                          x0=x, X_orig=X_orig, denoiser=denoiser,iter_max=iter_max2,sigma = sigma2,**args) 

    return x, psnr, ssim, psnrall


def gap_joint_denoise(y, Phi_sum, A, At, x0=None, X_orig=None, denoiser='tv+ffdnet', 
                      iter_max1=50, iter_max2=50, sigma1 = None, sigma2=None, **args):
    '''
    GAP-based joint denoise: multi period and multi step denoise
    
    '''
    # [1] 1st period denoising: gaptv_denoising
    print('*** 1st period denoising ***')
    x, psnr, ssim, psnrall =  gap_denoise(y, Phi_sum, A, At,x0=x0, 
                                          X_orig=X_orig, denoiser='tv',iter_max=iter_max1,sigma = sigma1, **args)    
    
    # [2] 2nd period denoising: gap tv+ffdnet multistep denoising
    print('*** 2nd period denoising ***')
    x, psnr, ssim, psnrall =  gap_multistep_denoise(y, Phi_sum, A, At, x0=x, 
                                          X_orig=X_orig, denoiser=denoiser,iter_max=iter_max2,sigma = sigma2,**args) 

    return x, psnr, ssim, psnrall

def admm_multistep_denoise(y, Phi_sum, A, At, _lambda=1, gamma=0.0, accelerate=None,
                denoiser='tv', iter_max=50, noise_estimate=False, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, model=None, X_orig=None, show_iqa=True, tvm='tv_chambolle'):
    '''
    ADMM-based multistep denoise

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    denoiser : string, optional
        Denoiser used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, 
           "Distributed Optimization and Statistical Learning via the 
           Alternating Direction Method of Multipliers," Foundations and 
           Trends® in Machine Learning, vol. 3, no. 1, pp. 1-122, 2011.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.

    Code credit
    -----------
    Xin Yuan, Bell Labs, xyuan@bell-labs.com, created Aug 7, 2018.
    Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, 
      updated Jan 22, 2019.

    See Also
    --------
    gap_denoise
    '''
    # [0] initialization
    if x0 is None:
        x0 = At(y) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    # [1] start iteration for reconstruction
    x = x0 # initialization
    theta = x0
    b = np.zeros_like(x0)
    psnr_all = []
    k = 0
    time_start = time.time() # timing
    print('---> gap_multistep_denoise')
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            # Euclidean projection
            yb = A(theta+b)
            x = (theta+b) + _lambda*(At((y-yb)/(Phi_sum+gamma))) # ADMM 
            # switch denoiser 
            if denoiser.lower() == 'tv+ffdnet': # total variation (TV) + ffdnet denoising
                if idx== 0 and it==0:
                    print(' --- tv+ffdnet_denoising ---')
                    
                # [.0] pre-denoise
                # x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                # [.1] denoise_step1: tv denoising
                try:
                    if tvm == 'tv_chambolle':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'tv_bregman':
                        theta = denoise_tv_bregman(x-b, tv_weight, max_iter=tv_iter_max)
                    elif tvm == 'ITV3D_FGP':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'ITV2D_cham':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)                  
                    else:
                        raise TypeError("no such tv denoiser")
                except TypeError as e:
                    print("Exception: ",repr(e))
                  
                # [.2] denoise_step2: ffdnet_denoising
                theta = ffdnet_vdenoiser(theta, nsig, model)
                # theta = ffdnet_vdenoiser(x-b, nsig, model)     
                            
            elif denoiser.lower() == 'tv+fastdvdnet':
                if idx== 0 and it==0:
                    print(' --- tv+fastdvdnet_denoising ---')
                # [.0] pre-denoise
                # x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)                
                # [.1] denoise_step1: tv denoising
                try:
                    if tvm == 'tv_chambolle':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'tv_bregman':
                        theta = denoise_tv_bregman(x-b, tv_weight, max_iter=tv_iter_max)
                    elif tvm == 'ITV3D_FGP':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'ITV2D_cham':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)                  
                    else:
                        raise TypeError("no such tv denoiser")
                except TypeError as e:
                    print("Exception: ",repr(e))
                  
                # [.2] denoise_step2: ffdnet_denoising
                theta = fastdvdnet_denoiser(theta, nsig, model, gray=True)
                # theta = fastdvdnet_denoiser(x-b, nsig, model, gray=True)                  
                    
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            
            theta = np.clip(theta,0,1) # [zzh]  this is optional, sometimes, when you are sure that theta \in [0 1], you can use this to compress the noise
            
            b = b - (x-theta) # update residual
            
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                              'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  ADMM-{0} iteration {1: 3d}, ' 
                              'PSNR {2: 2.2f} dB.'.format(denoiser.upper(), 
                               k+1, psnr_all[k]))
            k = k+1
        time_now = time.time()
        print('----> finish {}/{} time cost {:.2f} min'.format(idx+1, len(sigma),(time_now-time_start)/60))     
    psnr_ = []
    ssim_ = []
    nmask = x.shape[2]
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
    return x, psnr_, ssim_, psnr_all



def gap_multistep_denoise(y, Phi_sum, A, At, _lambda=1, gamma=None, accelerate=True, 
                denoiser='tv+ffdnet', iter_max=50, noise_estimate=False, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                X_orig=None, model=None, show_iqa=True, tvm='tv_chambolle'):
    '''
    GAP-based multistep denoise

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray,
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    denoiser : string, optional
        Denoiser used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.
    model : pretrained model for image/video denoising.
    tvm : string, optional, {'tv_chambolle', 'ATV_ClipA', 'ATV_ClipB','ATV_cham','ATV_FGP',
        'ITV2D_cham','ITV2D_FGP','ITV3D_cham','ITV3D_FGP'}
        tv denoiser type, default value = 'tv_chambolle' (zzh)

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] X. Liao, H. Li, and L. Carin, "Generalized Alternating Projection 
           for Weighted-$\ell_{2,1}$ Minimization with Applications to 
           Model-Based Compressive Sensing," SIAM Journal on Imaging Sciences, 
           vol. 7, no. 2, pp. 797-823, 2014.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.

    Code credit
    -----------
    Xin Yuan, Bell Labs, xyuan@bell-labs.com, created Aug 7, 2018.
    Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, 
      updated Jan 22, 2019.

    See Also
    --------
    admm_denoise
    '''
    # [0] initialization
    if x0 is None:
        # x0 = At(y, Phi) # default start point (initialized value)
        x0 = At(y) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    # y1 = np.zeros(y.shape)
    y1 = np.zeros_like(y) 
    # [1] start iteration for reconstruction
    x = x0 # initialization
    psnr_all = []
    k = 0
    time_start = time.time() # timing
    print('---> gap_multistep_denoise')
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            yb = A(x)
            if accelerate: # accelerated version of GAP
                y1 = y1 + (y-yb)
                x = x + _lambda*(At((y1-yb)/Phi_sum)) # GAP_acc
            else:
                x = x + _lambda*(At((y-yb)/Phi_sum)) # GAP
            # switch denoiser 
            if denoiser.lower() == 'tv+ffdnet': # total variation (TV) + ffdnet denoising
                if idx== 0 and it==0:
                    print(' --- tv+ffdnet_denoising ---')
                    
                # [.0] pre-denoise
                # x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                # [.1] denoise_step1: tv denoising
                try:
                    if tvm == 'tv_chambolle':
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'tv_bregman':
                        x = denoise_tv_bregman(x, tv_weight, max_iter=tv_iter_max)
                    elif tvm == 'ITV3D_FGP':
                        x = denoise_tv_FGP_ITV3D(x, tv_weight, n_iter_max=tv_iter_max)
                    elif tvm == 'ITV2D_cham':
                        x = denoise_tv_cham_ITV2D(x, tv_weight, n_iter_max=tv_iter_max)                        
                    else:
                        raise TypeError("no such tv denoiser")
                except TypeError as e:
                    print("Exception: ",repr(e))
                # [.2] denoise_step2: ffdnet_denoising
                x = ffdnet_vdenoiser(x, nsig, model)
                  
                    
            elif denoiser.lower() == 'tv+fastdvdnet':
                if idx== 0 and it==0:
                    print(' --- tv+fastdvdnet_denoising ---')
                    
                # [.0] pre-denoise
                # x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)                
                # [.1] denoise_step1: tv denoising
                try:
                    if tvm == 'tv_chambolle':
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'tv_bregman':
                        x = denoise_tv_bregman(x, tv_weight, max_iter=tv_iter_max)
                    elif tvm == 'ITV3D_FGP':
                        x = denoise_tv_FGP_ITV3D(x, tv_weight, n_iter_max=tv_iter_max)
                    elif tvm == 'ITV2D_cham':
                        x = denoise_tv_cham_ITV2D(x, tv_weight, n_iter_max=tv_iter_max)                        
                    else:
                        raise TypeError("no such tv denoiser")
                except TypeError as e:
                    print("Exception: ",repr(e))  
                    
                # [.2] denoise_step2: ffdnet_denoising
                x = fastdvdnet_denoiser(x, nsig, model, gray=True)
                
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  GAP-{0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, psnr_all[k]))
            k = k+1
        time_now = time.time()
        print('----> finish {}/{} time cost {:.2f} min'.format(idx+1, len(sigma),(time_now-time_start)/60))        
    psnr_ = []
    ssim_ = []
    nmask = x.shape[2]
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
    return x, psnr_, ssim_, psnr_all


def admm_denoise(y, Phi_sum, A, At, _lambda=1, gamma=0.0, accelerate=None,
                denoiser='tv', iter_max=50, noise_estimate=False, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, model=None, X_orig=None, show_iqa=True, tvm='tv_chambolle'):
    '''
    Alternating direction method of multipliers (ADMM)[1]-based denoising 
    regularization for snapshot compressive imaging (SCI).

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    denoiser : string, optional
        Denoiser used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, 
           "Distributed Optimization and Statistical Learning via the 
           Alternating Direction Method of Multipliers," Foundations and 
           Trends® in Machine Learning, vol. 3, no. 1, pp. 1-122, 2011.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.

    Code credit
    -----------
    Xin Yuan, Bell Labs, xyuan@bell-labs.com, created Aug 7, 2018.
    Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, 
      updated Jan 22, 2019.

    See Also
    --------
    gap_denoise
    '''
    # [0] initialization
    if x0 is None:
        x0 = At(y) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    # [1] start iteration for reconstruction
    x = x0 # initialization
    theta = x0
    b = np.zeros_like(x0)
    psnr_all = []
    k = 0
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            # Euclidean projection
            yb = A(theta+b)
            x = (theta+b) + _lambda*(At((y-yb)/(Phi_sum+gamma))) # ADMM
            # switch denoiser 
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                try:
                    if tvm == 'tv_chambolle':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'ITV3D_FGP':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'ITV2D_cham':
                        theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)                   
                    else:
                        raise TypeError("no such tv denoiser")
                except TypeError as e:
                    print("Exception: ",repr(e))
            elif denoiser.lower() == 'wavelet': # wavelet denoising
                if noise_estimate or nsig is None: # noise estimation enabled
                    theta = denoise_wavelet(x-b, multichannel=multichannel)
                else:
                    theta = denoise_wavelet(x-b, sigma=nsig, multichannel=multichannel)
            # elif denoiser.lower() == 'vnlnet': # Video Non-local net denoising
            #     theta = vnlnet(np.expand_dims((x-b).transpose(2,0,1),3), nsig)
            #     theta = np.transpose(theta.squeeze(3),(1,2,0))
            elif denoiser.lower() == 'ffdnet': # FFDNet frame-wise video denoising
                # x = ffdnet_vdenoiser(x, nsig, model)                  # [zzh] original
                theta = ffdnet_vdenoiser(x-b, nsig, model)              # [zzh] new code from xinyuan(1/3)
            elif denoiser.lower() == 'fastdvdnet': # FastDVDnet video denoising
                # x = fastdvdnet_denoiser(x, nsig, model, gray=True)    # [zzh] original
                theta = fastdvdnet_denoiser(x-b, nsig, model, gray=True) # [zzh] new code from xinyuan(2/3)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            
            theta = np.clip(theta,0,1) # [zzh] new code from xinyuan(3/3)
            
            b = b - (x-theta) # update residual
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                              'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  ADMM-{0} iteration {1: 3d}, ' 
                              'PSNR {2: 2.2f} dB.'.format(denoiser.upper(), 
                               k+1, psnr_all[k]))
            k = k+1
    
    psnr_ = []
    ssim_ = []
    nmask = x.shape[2]
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
    return x, psnr_, ssim_, psnr_all


def gap_denoise(y, Phi_sum, A, At, _lambda=1, gamma=None, accelerate=True, 
                denoiser='tv', iter_max=50, noise_estimate=False, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                X_orig=None, model=None, show_iqa=True, tvm='tv_chambolle'):
    '''
    Alternating direction method of multipliers (ADMM)[1]-based denoising 
    regularization for snapshot compressive imaging (SCI).

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray,
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    denoiser : string, optional
        Denoiser used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.
    model : pretrained model for image/video denoising.
    tvm : string, optional, {'tv_chambolle', 'ATV_ClipA', 'ATV_ClipB','ATV_cham','ATV_FGP',
        'ITV2D_cham','ITV2D_FGP','ITV3D_cham','ITV3D_FGP'}
        tv denoiser type, default value = 'tv_chambolle' (zzh)

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] X. Liao, H. Li, and L. Carin, "Generalized Alternating Projection 
           for Weighted-$\ell_{2,1}$ Minimization with Applications to 
           Model-Based Compressive Sensing," SIAM Journal on Imaging Sciences, 
           vol. 7, no. 2, pp. 797-823, 2014.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.

    Code credit
    -----------
    Xin Yuan, Bell Labs, xyuan@bell-labs.com, created Aug 7, 2018.
    Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, 
      updated Jan 22, 2019.

    See Also
    --------
    admm_denoise
    '''
    # [0] initialization
    if x0 is None:
        # x0 = At(y, Phi) # default start point (initialized value)
        x0 = At(y) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    # y1 = np.zeros(y.shape)
    y1 = np.zeros_like(y) 
    # [1] start iteration for reconstruction
    x = x0 # initialization
    psnr_all = []
    k = 0
    time_start = time.time() # timing
    print('---> gap_denoise')
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            yb = A(x)
            if accelerate: # accelerated version of GAP
                y1 = y1 + (y-yb)
                x = x + _lambda*(At((y1-yb)/Phi_sum)) # GAP_acc
            else:
                x = x + _lambda*(At((y-yb)/Phi_sum)) # GAP
            # switch denoiser 
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                try:
                    if tvm == 'tv_chambolle':
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
                    elif tvm == 'ITV3D_FGP':
                        x = denoise_tv_FGP_ITV3D(x, tv_weight, n_iter_max=tv_iter_max)
                    elif tvm == 'ITV2D_cham':
                        x = denoise_tv_cham_ITV2D(x, tv_weight, n_iter_max=tv_iter_max)                        
                    else:
                        raise TypeError("no such tv denoiser")
                except TypeError as e:
                    print("Exception: ",repr(e))
                    
            elif denoiser.lower() == 'wavelet': # wavelet denoising
                if noise_estimate or nsig is None: # noise estimation enabled
                    x = denoise_wavelet(x, multichannel=multichannel)
                else:
                    x = denoise_wavelet(x, sigma=nsig, multichannel=multichannel)
            # elif denoiser.lower() == 'vnlnet': # Video Non-local net denoising
            #     x = vnlnet(np.expand_dims(x.transpose(2,0,1),3), nsig)
            #     x = np.transpose(x.squeeze(3),(1,2,0))
            elif denoiser.lower() == 'ffdnet': # FFDNet frame-wise video denoising
                x = ffdnet_vdenoiser(x, nsig, model)
            elif denoiser.lower() == 'fastdvdnet': # FastDVDnet video denoising
                x = fastdvdnet_denoiser(x, nsig, model, gray=True)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  GAP-{0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, psnr_all[k]))
            k = k+1
        time_now = time.time()
        print('----> finish {}/{} time cost {:.2f} min'.format(idx+1, len(sigma),(time_now-time_start)/60))
            
    psnr_ = []
    ssim_ = []
    nmask = x.shape[2]
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
    return x, psnr_, ssim_, psnr_all



def GAP_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, X_ori):
    y1 = np.zeros((row,col))
    begin_time = time.time()
    f = At(y,Phi)
    for ni in range(maxiter):
        fb = A(f,Phi)
        y1 = y1+ (y-fb)
        f  = f + np.multiply(step_size, At( np.divide(y1-fb,Phi_sum),Phi ))
        f = denoise_tv_chambolle(f, weight,n_iter_max=30,multichannel=True)
    
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(f,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("GAP-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(f, X_ori), end_time-begin_time))
    return f

def ADMM_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, eta,X_ori):
    #y1 = np.zeros((row,col))
    begin_time = time.time()
    theta = At(y,Phi)
    v =theta
    b = np.zeros((row,col,ColT))
    for ni in range(maxiter):
        yb = A(theta+b,Phi)
        #y1 = y1+ (y-fb)
        v  = (theta+b) + np.multiply(step_size, At( np.divide(y-yb,Phi_sum+eta),Phi ))
        #vmb = v-b
        theta = denoise_tv_chambolle(v-b, weight,n_iter_max=30,multichannel=True)
        
        b = b-(v-theta)
        weight = 0.999*weight
        eta = 0.998 * eta
        
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(v,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("ADMM-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(v, X_ori), end_time-begin_time))
    return v
