
from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
# import matplotlib.pyplot as plt # [for debug]

## load 'orig' & 'mask' and create 'meas' to form a dataset
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


## load all 'orig' & 'mask' and create 'meas' to form a dataset in the memory
class AllOrigTrainDataset(Dataset):

    def __init__(self, orig_train_path, mask_full_path):
        super(AllOrigTrainDataset, self).__init__()
        # get orig data paths
        orig_train_full_path = []
        if os.path.exists(orig_train_path):
            orig_train_list = os.listdir(orig_train_path)
            orig_train_full_path = [orig_train_path + '/' + orig_train_list[i] for i in range(len(orig_train_list))]
            self.orig_len = len(orig_train_list)
        else:
            raise FileNotFoundError('orig_train_path doesn\'t exist!')
        
        # load mask
        if os.path.exists(mask_full_path):
            mask = scio.loadmat(mask_full_path)
            # data type convert
            mask = torch.from_numpy(mask['mask'])
            mask = mask.float()
         # rescale to 0-1
            mask_maxv = torch.max(mask)
            if mask_maxv > 1:
                mask = torch.div(mask, mask_maxv)           
        else:
            raise FileNotFoundError('mask_full_path doesn\'t exist!')
        
        # load all orig/gt and calc meas
        img_sz = mask.numpy().shape[0:2]
        Cr = mask.numpy().shape[2]
        print('img_sz: {} Cr: {}'.format(img_sz, Cr)) # for debug
        self.origs = torch.zeros([*img_sz,Cr,self.orig_len],dtype=torch.float)
        self.meass = torch.zeros([*img_sz,self.orig_len],dtype=torch.float)

        print('orig images loading...')
        for i in range(self.orig_len):
            orig_train_ith_path = orig_train_full_path[i]

            gt = scio.loadmat(orig_train_ith_path)
            
            if "patch_save" in gt:
                gt = torch.from_numpy(gt['patch_save'] / 255)
            elif 'orig' in gt:
                gt = torch.from_numpy(gt['orig'] / 255)
            # elif "p1" in gt:
            #     gt = torch.from_numpy(gt['p1'] / 255)
            # elif "p2" in gt:
            #     gt = torch.from_numpy(gt['p2'] / 255)
            # elif "p3" in gt:
            #     gt = torch.from_numpy(gt['p3'] / 255)
            
            # data type convert
            gt = gt.float()
                
            # [debug] dtype info and imshow
            # print('gt dtype:{}, meas dtype:{}'.format(gt.dtype, mask.dtype))
            # plt.imshow(gt[:,:,1].numpy())
            # plt.show()

            # calculate meas
            meas = torch.sum(torch.mul(mask, gt),2)

            # save to the full date array
            self.origs[...,i] = gt
            self.meass[...,i] = meas
            if i%500==0:
                print('{:.2f}% finished'.format(i*100/self.orig_len))
        # [debug] shape info
        print('orig images loaded!')
        print('origs shape:{}, meass shape:{}'.format(self.origs.shape, self.meass.shape))

            
    def __getitem__(self, index):
        gt = self.origs[:,:,:,index]
        meas = self.meass[:,:,index]
        # permute
        gt = gt.permute(2, 0, 1)
        return gt, meas

    def __len__(self):
        return self.orig_len


## directly load 'orig/gt' & 'meas' to form a dataset
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

## directly load 'orig/gt' & 'meas' to form a dataset