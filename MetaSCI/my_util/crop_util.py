import numpy as np
import cv2
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from einops import rearrange
class ImgPatches():
    def __init__(self,img,ps_h,ps_w,overlap) -> None:
        H,W,C=img.shape
        self.h=H
        self.w=W
        self.c=C
        self.ps_h = ps_h
        self.ps_w = ps_w
        overlap_h = int(ps_h*overlap)
        overlap_w = int(ps_w*overlap)
        self.step_h = ps_h-int(overlap*ps_h)
        self.step_w = ps_w-int(overlap*ps_w)
        self.chunk_h = (H-overlap_h)//(ps_h-overlap_h)*(ps_h-overlap_h)+overlap_h
        self.chunk_w = (W-overlap_w)//(ps_w-overlap_w)*(ps_w-overlap_w)+overlap_w
    def crop(self,img,batchsize=1):
        chunk_main = img[:self.chunk_h,:self.chunk_w,:]
        chunk_right = img[:self.chunk_h,-self.ps_w:,:]
        chunk_lower = img[-self.ps_h:,:self.chunk_w,:]
        chunk_corner = img[-self.ps_h:,-self.ps_w:,:]
        self.chunkshape=[chunk_main.shape,chunk_right.shape,chunk_lower.shape]
        chunk_main_patches = patchify(chunk_main, (self.ps_h,self.ps_w,self.c), 
                    step=(self.step_h,self.step_w,1))
        chunk_right_patches = patchify(chunk_right, (self.ps_h,self.ps_w,self.c), 
                    step=(self.step_h,self.step_w,1))
        chunk_lower_patches = patchify(chunk_lower, (self.ps_h,self.ps_w,self.c), 
                    step=(self.step_h,self.step_w,1))
        self.chunkcountshape=[chunk_main_patches.shape[:3],chunk_right_patches.shape[:3],
                        chunk_lower_patches.shape[:3]]
        chunk_main_patches = rearrange(chunk_main_patches,'p1 p2 p3 s1 s2 s3 -> (p1 p2 p3) s1 s2 s3')
        chunk_right_patches = rearrange(chunk_right_patches,'p1 p2 p3 s1 s2 s3 -> (p1 p2 p3) s1 s2 s3')
        chunk_lower_patches = rearrange(chunk_lower_patches,'p1 p2 p3 s1 s2 s3 -> (p1 p2 p3) s1 s2 s3')
        chunk_corner_patches = rearrange(chunk_corner,' s1 s2 s3 -> () s1 s2 s3')
        self.patchcount=[chunk_main_patches.shape[0],chunk_right_patches.shape[0],
                        chunk_lower_patches.shape[0]]
        patches = np.concatenate([chunk_main_patches,chunk_right_patches,
                                chunk_lower_patches,chunk_corner_patches],0)
        patches_count = patches.shape[0]
        patches_batchlist=[]
        if patches_count<=batchsize:
            return [patches]
        else:
            for batch_index in range(int(np.ceil(patches_count/batchsize))-1):
                patches_batchlist.append(patches[batch_index*batchsize:(batch_index+1)*batchsize])
            patches_batchlist.append(patches[(batch_index+1)*batchsize:])
            return patches_batchlist
    def merge(self,patches:list):
        patches=np.concatenate(patches,0)
        chunk_main_patches = patches[:self.patchcount[0]]
        chunk_right_patches = patches[self.patchcount[0]:self.patchcount[0]+self.patchcount[1]]
        chunk_lower_patches = patches[self.patchcount[0]+self.patchcount[1]:self.patchcount[0]+self.patchcount[1]+self.patchcount[2]]
        chunk_corner_patches = patches[-1]
        chunk_main_patches = rearrange(chunk_main_patches,'(p1 p2 p3) s1 s2 s3 -> p1 p2 p3 s1 s2 s3',
                    p1=self.chunkcountshape[0][0],p2=self.chunkcountshape[0][1],p3=self.chunkcountshape[0][2])
        chunk_right_patches = rearrange(chunk_right_patches,'(p1 p2 p3) s1 s2 s3 -> p1 p2 p3 s1 s2 s3',
                    p1=self.chunkcountshape[1][0],p2=self.chunkcountshape[1][1],p3=self.chunkcountshape[1][2])
        chunk_lower_patches = rearrange(chunk_lower_patches,'(p1 p2 p3) s1 s2 s3 -> p1 p2 p3 s1 s2 s3',
                    p1=self.chunkcountshape[2][0],p2=self.chunkcountshape[2][1],p3=self.chunkcountshape[2][2])
        chunk_corner_patches = chunk_corner_patches
        chunk_main = unpatchify(chunk_main_patches,self.chunkshape[0])
        chunk_right = unpatchify(chunk_right_patches,self.chunkshape[1])
        chunk_lower = unpatchify(chunk_lower_patches,self.chunkshape[2])
        chunk_corner = chunk_corner_patches
        #
        img=np.zeros((self.h,self.w,self.c),dtype=np.float)
        img[:self.chunk_h,:self.chunk_w,:]+=chunk_main
        img[:self.chunk_h,-self.ps_w:,:]+=chunk_right
        img[-self.ps_h:,:self.chunk_w,:]+=chunk_lower
        img[-self.ps_h:,-self.ps_w:,:]+=chunk_corner
        #
        weight = np.zeros((self.h,self.w,self.c),dtype=np.float)
        weight[:self.chunk_h,:self.chunk_w,:]+=np.ones_like(chunk_main)
        weight[:self.chunk_h,-self.ps_w:,:]+=np.ones_like(chunk_right)
        weight[-self.ps_h:,:self.chunk_w,:]+=np.ones_like(chunk_lower)
        weight[-self.ps_h:,-self.ps_w:,:]+=np.ones_like(chunk_corner)
        img = img / weight
        return img
if __name__=='__main__':
    path=r'C:\Users\yrz\Pictures\Lenna.png'
    img=cv2.imread(path,-1)[:92,:92,::-1]
    croppatch=ImgPatches(img,48,48,0.1)
    patches=croppatch.crop(img,32)
    img_merge = croppatch.merge(patches)
    fig,ax=plt.subplots(1,3)
    ax[0].imshow(img)
    ax[1].imshow(img_merge/255)
    ax[2].imshow(img_merge-img)
    plt.show()