'''
Test the Video Non-local Denoising Network (VNLnet).
'''
import os.path
import skvideo.io  # video reader and writer
from skimage.restoration import denoise_tv_chambolle  # TV denoiser

from .test import vnlnet

datname = 'traffic' # dataset name
sigma = 20 # noise standard deviation (sigma)
cur_dir = os.path.dirname(__file__)
vpath = '{}/input/{}_sigma{:d}.avi'.format(cur_dir,datname,sigma) # path to the noisy video
outpath = '{}/output/{}_sigma{:d}_vnlnet.avi'.format(cur_dir,datname,sigma) # path to the denoised video
# read video from file as ndarray
video = skvideo.io.vread(vpath, as_grey=True)
# apply VNLnet denoising for the noisy video
vdenoised = vnlnet(video, sigma)
# save the denoised video 
skvideo.io.vwrite(outpath, vdenoised)
