"""
@author : Ziheng Cheng, Bo Chen
@Email : zhcheng@stu.xidian.edu.cn      bchen@mail.xidian.edu.cn

Description:
    This is the data generating code for Snapshot Compressive Imaging reconstruction in recurrent convolution neural network

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

from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
# import matplotlib.pyplot as plt # [for debug]


class OrigTrainDataset(Dataset):

    def __init__(self, orig_train_path, mask_full_path):
        super(OrigTrainDataset, self).__init__()
        # get data paths
        self.orig_train_path = []
        self.mask_full_path = []
        if os.path.exists(orig_train_path):
            orig_train_list = os.listdir(orig_train_path)
            self.orig_train_path = [orig_train_path + '/' + orig_train_list[i] for i in range(len(orig_train_list))]

        else:
            raise FileNotFoundError('orig_train_path doesn\'t exist!')
        
        # get mask paths
        if os.path.exists(mask_full_path):
            self.mask_full_path = mask_full_path
        else:
            raise FileNotFoundError('mask_full_path doesn\'t exist!')
        

    def __getitem__(self, index):

        orig_train_path = self.orig_train_path[index]
        mask_full_path = self.mask_full_path
        
        # load orig and mask
        gt = scio.loadmat(orig_train_path)
        mask = scio.loadmat(mask_full_path)
        
        if "patch_save" in gt:
            gt = torch.from_numpy(gt['patch_save'] / 255)
        elif 'orig' in gt:
            gt = torch.from_numpy(gt['orig'] / 255)
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)

        mask = torch.from_numpy(mask['mask'])
        
        # data type convert
        mask = mask.float()
        gt = gt.float()
               
        # rescale to 0-1
        mask_maxv = torch.max(mask)
        if mask_maxv > 1:
            mask = torch.div(mask, mask_maxv)
        
        # [debug] dtype info and imshow
        # print('gt dtype:{}, meas dtype:{}'.format(gt.dtype, mask.dtype))
        # plt.imshow(gt[:,:,1].numpy())
        # plt.show()

        # calculate meas
        meas = torch.sum(torch.mul(mask, gt),2)
        # meas = torch.from_numpy(meas['meas'] / 255)

        # permute
        gt = gt.permute(2, 0, 1)

        # [debug] shape info
        # print('gt shape:{}, meas shape:{}'.format(gt.shape, meas.shape))

        return gt, meas

    def __len__(self):

        return len(self.orig_train_path)


class MeasTrainDataset(Dataset):
    def __init__(self, path):
        super(MeasTrainDataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            dir_list = os.listdir(path)
            groung_truth_path = path + '/gt'
            measurement_path = path + '/measurement'

            if os.path.exists(groung_truth_path) and os.path.exists(measurement_path):
                groung_truth = os.listdir(groung_truth_path)
                measurement = os.listdir(measurement_path)
                self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i],
                              'measurement': measurement_path + '/' + measurement[i]} for i in range(len(groung_truth))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        groung_truth, measurement = self.data[index]["groung_truth"], self.data[index]["measurement"]

        gt = scio.loadmat(groung_truth)
        meas = scio.loadmat(measurement)
        if "patch_save" in gt:
            gt = torch.from_numpy(gt['patch_save'] / 255)
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)

        meas = torch.from_numpy(meas['meas'] / 255)

        gt = gt.permute(2, 0, 1)

        # print(tran(img).shape)

        return gt, meas

    def __len__(self):

        return len(self.data)
