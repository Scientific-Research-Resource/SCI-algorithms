from __future__ import division
import numpy as np
import h5py
import scipy.io as sio
import hdf5storage
import random
import matplotlib.pyplot as plt
import math
import os


def Data_Division(dataset_name,nF, is_testing=False, experiment=True):
    """
    :param dataset_name: the common part in the name of the dataset
    :return: 
        Dataset tuple: pair_train, pair_test, pair_valid (measurement, ground-truth)
        Sensing Model: mask pre-modeled according the optical structure
    """

    # (data_name,mask_name),file_id_list,file_id_valid_list,file_id_test_list= dataset_name,[],[],[]
    (data_name,mask_name)= dataset_name

    if is_testing is True:
        # for meas test
        label_training, label_valid = [], []  # ignore
    else:
        # label_training = os.listdir(data_name[0])
        label_training = [file_x for file_x in os.listdir(data_name[0]) if file_x.endswith('.mat')] # get .mat files
        # label_valid = os.listdir(data_name[1])
        label_valid = [file_x for file_x in os.listdir(data_name[1]) if file_x.endswith('.mat')]
        label_valid.sort()

    # label_testing = os.listdir(data_name[2]) # zzh
    label_testing = [file_x for file_x in os.listdir(data_name[2]) if file_x.endswith('.mat')]
    label_testing.sort()
    #sample_num = len(label_training)
    #sample_num_valid = len(label_valid)
    #sample_num_test = len(label_testing)
    
    mask_file = sio.loadmat(mask_name+'.mat') # for '-v7' .mat file (MATLAB)
    mask = mask_file['mask']
    # with h5py.File(mask_name+'.mat', 'r') as mask_file: # for '-v7.3' .mat file (MATLAB), test it before use-zzh
    #     mask = np.array(mask_file['mask']) 
    #     mask.transpose((1, 2, 0))         
    # all zero place change into 1
    p = np.argwhere(np.sum(mask,2)==0)
    np.random.seed(1)
    d_zero = np.random.randint(nF,size=len(p))
    for i in range(len(p)):
        mask[p[i][0],p[i][1],d_zero[i]] = 1
 
    return label_training, label_valid, label_testing, mask

# def Data_Generator_File(dataset_name, label, mask, batch_size, nF, is_training=True,is_valid=False,is_testing=False):
def Data_Generator_File(dataset_name, label, mask, batch_size, nF, is_training=True,is_valid=False,is_testing=False, is_testing_meas=False):
    (data_name,mask_name) = dataset_name
    if is_testing_meas is False:
        key_name = 'patch_save'
    else:
        key_name = 'meas'
        
    # W, H ,nC= 256,256,nF
    sample_num = len(label)
    index = np.random.choice(sample_num, size=sample_num, replace=False).astype(np.int16)
    sample_cnt,batch_cnt,list_measure,list_ground = 0,0,[],[]
    while True:
        if (sample_cnt < sample_num):
            if is_training is True:
                ind_set = index[sample_cnt]
            else:
                ind_set = sample_cnt
            if is_testing is False:
                if is_valid is False:
                    img = sio.loadmat(data_name[0]+label[ind_set])                      #['patch_save']
                    if "patch_save" in img:
                        img = img[key_name]
                    elif "p1" in img:
                        img = img['p1']
                    elif "p2" in img:
                        img = img['p2']
                    elif "p3" in img:
                        img = img['p3']
                else:
                    img = sio.loadmat(data_name[1]+label[ind_set])[key_name]
            else:
                img = sio.loadmat(data_name[2]+label[ind_set])[key_name]


            if is_testing_meas is False:
                img=img/255
                meas = np.sum(mask*img,2) # input orig as train/valid/test pic
            elif (is_testing is True) and (is_testing_meas is True):
                meas = img # input meas as test pic
                if meas.max() > 50:
                    # meas is generated from orig with grayscale ranging 0-255, rescale it
                    meas = meas/255

            ### shot noise injection, for training for real data test
            # if (is_training is True) and (add_noise is True):
            #     QE, bit = 0.4, 512
            #     meas_max = meas.max()
            #     meas = meas/meas_max
            #     meas = np.random.binomial((meas*bit/QE).astype(int),QE)
            #     meas = meas/bit
            #     meas = meas*meas_max
              
            ### y/sum(phi) 
            C = np.sum(mask**2,2)
            C[C==0]=1
            meas = meas/C
            meas_temp = np.tile(meas[:,:,np.newaxis],(1,1,nF))
            meas_temp = meas_temp*mask
            
            
            list_measure.append(meas_temp)
            if is_testing_meas is False:
                truth_img = img
            else:
                # for testing_meas, there is no ground truth, substituded with all-zero array
                truth_img = np.zeros_like(mask)
            list_ground.append(truth_img)
            batch_cnt += 1
            sample_cnt += 1
                                      
            if batch_cnt == batch_size:
                batch_measure,batch_ground = np.stack(list_measure,0),np.stack(list_ground,0)
                height_init,batch_cnt,list_measure,list_ground = 0,0,[],[]
                #print batch_measure.shape,batch_ground.shape
                yield batch_measure,mask,batch_ground
                #print 'sample_index'+str(sample_index)
        else:            
            sample_cnt = 0
            index = np.random.choice(sample_num, size=sample_num, replace=False).astype(np.int16)
            


         
