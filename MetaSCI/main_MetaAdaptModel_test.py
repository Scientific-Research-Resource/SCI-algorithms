"""
@author : Zhihong Zhang
# Adaptation model test for metaSCI
# modified: Zhihong Zhang, 2021.6

Note:
- 

Todo:


"""
import numpy as np
from datetime import datetime
import os
import logging
from os.path import join as opj
from os.path import exists as ope
import random
import scipy.io as sci
from utils import generate_masks_MAML, generate_meas
from my_util.plot_util import plot_multi
from my_util.quality_util import cal_psnrssim
import time
from tqdm import tqdm
from MetaFunc import construct_weights_modulation, forward_modulation

# %% setting
# envir config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide tensorflow warning

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.reset_default_graph()


# params config
# setting global parameters
batch_size = 1
num_frame = 10
image_dim = 256
Epoch = 3
sigmaInit = 0.01
step = 1
update_lr = 1e-5
num_updates = 5
picked_task = list(range(1,5))  # pick masks for base model train
num_task = len(picked_task)  # num of picked masks
run_mode = 'test'  # 'train', 'test','finetune'
test_real = False  # test real data
pretrain_model_idx = -1  # pretrained model index, 0 for no pretrained
exp_name = "Realmask_AdaptTest_256_Cr10_zzhTest"
model_name_prefix = 'adapt_model'
timestamp = '{:%m-%d_%H-%M}'.format(datetime.now())  # date info

# data path
# trainning set
# datadir = "../[data]/dataset/testing_truth/bm_256_10f/"
datadir = "../[data]/benchmark/orig/bm_256/"
# datadir = "../[data]/dataset/testing_truth/test_256_10f/"
maskpath = "./dataset/mask/realMask_256_Cr10_N576_overlap50.mat"


# model path
# pretrain_model_path = './result/_pretrained_model/simulate_data_256_Cr8/'
pretrain_model_path = './result/train/M_RealmaskDemo_AdaptTrain_256_Cr10_zzhTest/trained_model/'

# saving path
save_path = './result/test/'+exp_name+'_'+timestamp+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# logging setting
logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # handler for console output
chlr.setFormatter(formatter)
chlr.setLevel('INFO')
fhlr = logging.FileHandler(save_path+'train.log')  # handler for log file
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info('\t Exp. name: '+exp_name)
logger.info('\t Mask path: '+maskpath)
logger.info('\t Data dir: '+datadir)
logger.info('\t pretrain model: '+pretrain_model_path)
logger.info('\t Params: batch_size {:d}, num_frame {:d}, image_dim {:d}, sigmaInit {:f}, update_lr {:f}, num_updates {:d}, picked_task {:s}, run_mode- {:s}, pretrain_model_idx {:d}'.format(
    batch_size, num_frame, image_dim, sigmaInit, update_lr, num_updates, str(picked_task), run_mode, pretrain_model_idx))

#%% construct graph, load pretrained params ==> train, finetune, test
weights, weights_m = construct_weights_modulation(sigmaInit,num_frame)

mask = tf.placeholder('float32', [image_dim, image_dim, num_frame])
meas_re = tf.placeholder('float32', [batch_size, image_dim, image_dim, 1])
gt = tf.placeholder('float32', [batch_size, image_dim, image_dim, num_frame])

final_output = forward_modulation(mask, meas_re, gt, weights, weights_m, batch_size, num_frame, image_dim)

nameList = os.listdir(datadir)
mask_sample, mask_s_sample = generate_masks_MAML(maskpath, picked_task)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for task_index in range(num_task):
        # load pretrained params
        ckpt = tf.train.get_checkpoint_state(pretrain_model_path+model_name_prefix+str(picked_task[task_index]))
        if ckpt:
            ckpt_states = ckpt.all_model_checkpoint_paths
            saver.restore(sess, ckpt_states[pretrain_model_idx]) 
            logger.info('===> Load pretrained model from: '+ckpt_states[pretrain_model_idx])
        else:
            logger.error('===> No pretrained model found')
            raise FileNotFoundError('No pretrained model found')
                                       
        # [==> test]                 
        logger.info('\n===== Task {:4d}/{:<4d} Test Begin=====\n'.format(task_index, len(picked_task)))
        validset_psnr = 0
        validset_ssim = 0  
        mask_sample_i = mask_sample[task_index]
        mask_s_sample_i = mask_s_sample[task_index]
        for index in tqdm(range(len(nameList))):
            # load data
            data_tmp = sci.loadmat(datadir + nameList[index])

            if test_real:
                gt_tmp = np.zeros([image_dim, image_dim, num_frame])
                assert "meas" in data_tmp, 'NotFound ERROR: No MEAS in dataset'
                meas_sample = data_tmp['meas'][task_index]
                # meas_tmp = data_tmp['meas']task_index / 255
            else:
                if "patch_save" in data_tmp:
                    gt_tmp = data_tmp['patch_save'] / 255
                elif "orig" in data_tmp:
                    gt_tmp = data_tmp['orig'] / 255
                else:
                    raise FileNotFoundError('No ORIG in dataset')           
                meas_sample,gt_sample = generate_meas(gt_tmp, mask_sample_i)
            
            # normalize data
            mask_max = np.max(mask_sample_i) 
            mask_sample_i = mask_sample_i/mask_max
            mask_s_sample_i = mask_s_sample_i/mask_max # to be verified
            
            meas_sample = meas_sample/mask_max
            meas_sample_re = meas_sample / mask_s_sample_i
            meas_sample_re = np.expand_dims(meas_sample_re, -1)

        
            # test data
            pred = np.zeros((image_dim, image_dim, num_frame,meas_sample_re.shape[0]))
            time_all = 0
            for k in range(meas_sample_re.shape[0]):
                meas_sample_re_k = np.expand_dims(meas_sample_re[k],0)
                gt_sample_k =  np.expand_dims(gt_sample[k],0)
                
                begin = time.time()
                pred_k = sess.run([final_output['pred']],
                        feed_dict={mask: mask_sample_i,
                                    meas_re: meas_sample_re_k,
                                    gt: gt_sample_k}) # pred for Y_meas
                time_all += time.time() - begin
                
                pred[...,k] = pred_k[0]
            
            
            
            # eval: psnr, ssim
            mean_psnr,mean_ssim = 0,0
            psnr_all = np.zeros(0)
            ssim_all = np.zeros(0)                 
            if np.sum(gt_sample)!=0:
                for m in range(meas_sample_re.shape[0]):
                    psnr_all_m = np.zeros(0)
                    ssim_all_m = np.zeros(0)
                    for k in range(num_frame):      
                        psnr_k, ssim_k = cal_psnrssim(gt_sample[m,...,k], pred[...,k,m])
                        psnr_all_m = np.append(psnr_all_m,psnr_k)
                        ssim_all_m =np.append(ssim_all_m,ssim_k)
                        
                    psnr_all = np.append(psnr_all,psnr_all_m)
                    ssim_all =np.append(ssim_all,ssim_all_m)
                    
                    # save image
                    plot_multi(pred[...,m], 'MeasRecon_Task%d_%s_Frame%d'%(picked_task[task_index], nameList[index].split('.')[0],m), col_num=num_frame//2, titles=psnr_all_m,savename='MeasRecon_Task%d_%s_Frame%d_psnr%.2f_ssim%.2f'%(picked_task[task_index], nameList[index].split('.')[0],m,np.mean(psnr_all_m),np.mean(ssim_all_m)), savedir=save_path+'recon_img/task%d/'%picked_task[task_index])                            
                                        
                mean_psnr = np.mean(psnr_all)
                mean_ssim = np.mean(ssim_all)
                
                validset_psnr += mean_psnr
                validset_ssim += mean_ssim  
                                    
                logger.info('---> Task {} - {:<20s} Recon complete: PSNR {:.2f}, SSIM {:.2f}, Time {:.2f}'.format(picked_task[task_index], nameList[index], mean_psnr, mean_ssim, time_all))

            mat_save_path = save_path+'recon_mat/task%d/'%picked_task[task_index]
            if not ope(mat_save_path):
                os.makedirs(mat_save_path)
            sci.savemat(mat_save_path+'MeasRecon_Task%d_%s_psnr%.2f_ssim%.2f.mat'%(picked_task[task_index], nameList[index].split('.')[0],mean_psnr,mean_ssim),
                        {'recon':pred, 
                        'gt':gt_sample,
                        'psnr_all':psnr_all,
                        'ssim_all':ssim_all,
                        'mean_psnr':mean_psnr,
                        'mean_ssim':mean_ssim,
                        'time_all':time_all,
                        'task_index':picked_task[task_index]            
                        })
            logger.info('---> Recon data saved to: '+save_path)
        validset_psnr /= len(nameList)
        validset_ssim /= len(nameList)       
        logger.info('===> Task {:4d}/{:<4d} Recon complete: Aver. PSNR {:.2f}, Aver.SSIM {:.2f}'.format(task_index,len(picked_task), validset_psnr, validset_ssim))