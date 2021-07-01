import numpy as np
import matplotlib.pylab as plt
import cv2
from numpy.lib.npyio import save
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from os.path import join as opj
from os.path import exists as ope
from os.path import dirname as opd
from tqdm import tqdm
def plot(img,title="",savename="",savedir=None):  
    # plot a figure  
    plt.figure()
    plt.title(title)
    plt.imshow(img,vmax=img.max(),vmin=0)
    if savedir!=None:
        if not ope(savedir):
            os.makedirs(savedir)        
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()
    
def plot12(img1,img2,title1="",title2="",title="",savename="",savedir=None):
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(121)
    plt.title(title1)
    plt.imshow(img1,vmax=img1.max(),vmin=0)
    plt.subplot(122)
    plt.title(title2)
    plt.imshow(img2,vmax=img2.max(),vmin=0)
    if savedir!=None:
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()
    
def plot_multi(imgs, title="", titles=None, col_num=4, fig_sz=None, savename="", savedir=None):
    # plot multiple figures
    fig = plt.figure()
    fig.suptitle(title)
    
    num_img = imgs.shape[-1]

    for nt in range(num_img):
        plt.subplot(np.ceil(num_img/col_num), col_num, nt+1)
        plt.imshow(imgs[...,nt], cmap=plt.cm.gray, vmin=0, vmax=imgs.max())
        plt.axis('off')
        if titles is not None:
            if isinstance(titles[0],str):
                # titles are strings
                plt.title('#{0:d}: '.format(nt+1) + titles[nt], fontsize=12)
            else:
                # titles are values
                plt.title('#{0:d}: {1:.2f}'.format(nt+1,  titles[nt]), fontsize=12)
    plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0, top=1, left=0, right=1)
    
    if savedir is not None:
        if not ope(savedir):
            os.makedirs(savedir)
        plt.savefig('{}{}.png'.format(savedir,savename),dpi=200) 
    else:
        plt.show()
    plt.close()

def plot_hist(array,bins=None,title='',savename="",savedir=None):
    plt.figure()
    plt.title(title)
    if bins!=None:
        plt.hist(array,bins=bins)
    else:
        plt.hist(array)
    if savedir!=None:
        if not ope(savedir):
            os.makedirs(savedir)        
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()
def plot_matrix(matrix,cmap='viridis_r',vmin=None,vmax=None,text=False,title='',savename="",savedir=None):
    plt.figure(figsize=(20,20))
    plt.title(title)
    if vmin!=None and vmax!=None:
        plt.imshow(matrix,cmap=cmap,vmin=vmin,vmax=vmax)
    else:
        plt.imshow(matrix,cmap=cmap)
    plt.colorbar(shrink=0.8)
    if text:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, "{:.2f}".format(matrix[i, j]),
                            ha="center", va="center", color="w",size=8)
    if savedir!=None:
        if not ope(savedir):
            os.makedirs(savedir)
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()
def plot_boxplot(array,showfliers=True,whis=1.5,flierprops=None,title='',savename="",savedir=None):
    plt.figure()
    plt.title(title)
    plt.boxplot(array,showfliers=showfliers,whis=whis,flierprops=flierprops)
    if savedir!=None:
        if not ope(savedir):
            os.makedirs(savedir)
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()

def plot12_boxplot(array1,array2,showfliers=True,whis=1.5,flierprops=None,
                    title1="",title2="",title="",savename="",savedir=None):
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(121)
    plt.title(title1)
    plt.boxplot(array1,showfliers=showfliers,whis=whis,flierprops=flierprops)
    plt.subplot(122)
    plt.title(title2)
    plt.boxplot(array2,showfliers=showfliers,whis=whis,flierprops=flierprops)
    if savedir!=None:
        if not ope(savedir):
            os.makedirs(savedir)
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()