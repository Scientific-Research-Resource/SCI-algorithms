"""
@author : Ziheng Cheng, Bo Chen
@Email : zhcheng@stu.xidian.edu.cn      bchen@mail.xidian.edu.cn

Description:


Citation:
    The code prepares for ECCV 2020

Contact:
    Ziheng Cheng
    zhcheng@stu.xidian.edu.cn
    Xidian University, Xi'an, China

    Bo Chen
    bchen@mail.xidian.edu.cn
    Xidian University, Xi'an, China

LICENSE
=======================================================================

The code is for research purpose only. All rights reserved.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

Copyright (c), 2020, Ziheng Cheng
zhcheng@stu.xidian.edu.cn

"""


import scipy.io as scio
import numpy as np


def generate_masks(mask_path):
    mask = scio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask_s = np.sum(mask, axis=2)
    index = np.where(mask_s == 0)
    mask_s[index] = 1

    return mask.astype(np.float32), mask_s.astype(np.float32)

# def generate_masks_metaTest(mask_path):
#     mask = scio.loadmat(mask_path + '/Mask.mat')
#     mask = mask['mask']
#     mask_s = np.sum(mask, axis=2)
#     index = np.where(mask_s == 0)
#     mask_s[index] = 1

#     mask = np.transpose(mask, [3, 0, 1, 2])
#     mask_s = np.transpose(mask_s, [2, 0, 1])

#     mask = mask[3]
#     mask_s = mask_s[3]

#     return mask.astype(np.float32), mask_s.astype(np.float32)

# def generate_masks_metaTest_v2(mask_path):
#     mask = scio.loadmat(mask_path + '/Mask.mat')
#     mask = mask['mask']
#     mask_s = np.sum(mask, axis=2)
#     index = np.where(mask_s == 0)
#     mask_s[index] = 1

#     mask = np.transpose(mask, [3, 0, 1, 2])
#     mask_s = np.transpose(mask_s, [2, 0, 1])

#     return mask.astype(np.float32), mask_s.astype(np.float32)

# def generate_masks_metaTest_v3(mask_path):
#     mask = scio.loadmat(mask_path)
#     mask = mask['mask']
#     mask_s = np.sum(mask, axis=2)
#     index = np.where(mask_s == 0)
#     mask_s[index] = 1

#     return mask.astype(np.float32), mask_s.astype(np.float32)

def generate_masks_MAML(mask_path, picked_task):
    # generate mask and mask_sum form given picked task index
    # input data: 
    #   mask_path->mask: [H,W,Cr,num_task]
    #   picked_task: list
    # output data: 
    #   mask: [num_task,H,W,Cr]
    #   mask_s: [num_task,H,W], mask sum
    mask = scio.loadmat(mask_path)
    mask = mask['mask']
    if mask.ndim==3:
        mask = np.expand_dims(mask,-1)
    mask_s = np.sum(mask, axis=2)
    index = np.where(mask_s == 0)
    mask_s[index] = 1

    mask = np.transpose(mask, [3, 0, 1, 2])
    mask_s = np.transpose(mask_s, [2, 0, 1])

    assert max(picked_task)<=mask.shape[0], 'ERROR: picked task index exceed maximum limit'
    mask = mask[picked_task]
    mask_s = mask_s[picked_task]
    return mask.astype(np.float32), mask_s.astype(np.float32)

def generate_meas(gt, mask):    
    """
    generate_meas [generate coded measurement from mask and orig] from mask and orig (extra orig frames with be throwed out)

    Args:
        gt [H,W,num_frame]: orig frames
        mask [H,W,Cr]: masks
    Returns:
        meas [num_batch,H,W]: coded measurement, each meas is a batch here
        used_gt [num_batch,H,W,Cr]: used orig frames
    """    
    # data type convert
    mask = mask.astype(np.float32)
    gt = gt.astype(np.float32)
    # rescale to 0-1
    mask_maxv = np.max(mask)
    if mask_maxv > 1:
        mask = mask/mask_maxv
        
    # calculate meas
    # meas = np.sum(mask*gt,2)
    Cr = mask.shape[2] # num of masks
    used_gt = np.zeros([gt.shape[2] // Cr, gt.shape[0], gt.shape[1], Cr])
    for jj in range(gt.shape[2] // Cr*Cr):
        if jj % Cr == 0:
            meas_t = np.zeros(gt.shape[0:2])
            n = 0
        pic_t = gt[:, :, jj]
        mask_t = mask[:, :, n]

        used_gt[jj // Cr, :, :, n] = pic_t
        n += 1
        meas_t = meas_t + np.multiply(mask_t, pic_t)

        if jj == Cr-1:
            meas_t = np.expand_dims(meas_t, 0)
            meas = meas_t
        elif (jj + 1) % Cr == 0: #zzh
            meas_t = np.expand_dims(meas_t, 0)
            meas = np.concatenate((meas, meas_t), axis=0)    
    return meas, used_gt
