import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def cal_psnrssim(img_gt,img_hat):
    """[summary]

    Args:
        img_gt ([type]): H W C
        img_hat ([type]): [description]

    Returns:
        [type]: [description]
    """
    # img_hat=img_hat*img_gt.mean()/img_hat.mean()
    # H,W,C=img_gt.shape
    img_hat_psnr=psnr(img_gt,img_hat,data_range=img_gt.max())
    img_hat_ssim = ssim(img_gt,img_hat,data_range=img_gt.max(),multichannel=True)
    return img_hat_psnr,img_hat_ssim
