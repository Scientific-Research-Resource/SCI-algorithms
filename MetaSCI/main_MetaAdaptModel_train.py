"""
@author : Zhihong Zhang
# Adaptation model training for metaSCI
# modified: Zhihong Zhang, 2021.6

Note:
- For now, each adaptation is based on the last adaption's params, as I think there's no need restore to the BaseModel's params

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
max_iter = 1000 # max iter in an epoch
picked_task = list(range(1,5))  # pick masks for base model train
num_task = len(picked_task)  # num of picked masks
run_mode = 'finetune'  # 'train', 'test','finetune'
test_real = False  # test real data
pretrain_model_idx = -1  # pretrained model index, 0 for no pretrained
exp_name = "Realmask_AdaptTrain_256_Cr10_zzhTest"
model_name_prefix = 'adapt_model'
timestamp = '{:%m-%d_%H-%M}'.format(datetime.now())  # date info

# data path
# trainning set
datadir = "../[data]/dataset/training_truth/data_augment_256_10f/"
valid_dir = "../[data]/dataset/testing_truth/bm_256_10f/"
maskpath = "./dataset/mask/realMask_256_Cr10_N576_overlap50.mat"


# model path
# pretrain_model_path = './result/_pretrained_model/simulate_data_256_Cr8/'
pretrain_model_path = './result/train/A_Realmask_BaseTrain_256_Cr10_06-18_19-25/trained_model/'

# saving path
save_path = './result/train/'+exp_name+'_'+timestamp+'/'
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
logger.info('\t model name prefix: '+model_name_prefix)
logger.info('\t Params: batch_size {:d}, num_frame {:d}, image_dim {:d}, sigmaInit {:f}, update_lr {:f}, num_updates {:d}, max_iter {:d}, picked_task {:s}, run_mode- {:s}, pretrain_model_idx {:d}'.format(
    batch_size, num_frame, image_dim, sigmaInit, update_lr, num_updates, max_iter, str(picked_task), run_mode, pretrain_model_idx))

# %% construct graph, load pretrained params ==> train, finetune, test
# Place holder
mask = tf.placeholder('float32', [image_dim, image_dim, num_frame])
meas_re = tf.placeholder('float32', [batch_size, image_dim, image_dim, 1])
gt = tf.placeholder('float32', [batch_size, image_dim, image_dim, num_frame])

# weights
weights, weights_m = construct_weights_modulation(sigmaInit, num_frame)

# feed forward
output = forward_modulation(mask, meas_re, gt, weights, weights_m, batch_size, num_frame, image_dim)
# optimize and save  weight_m
optimizer = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(output['loss'], var_list=list(weights_m.values()))
saver = tf.train.Saver()


# data names
nameList = os.listdir(datadir)
valid_nameList = os.listdir(valid_dir)
mask_sample, mask_s_sample = generate_masks_MAML(maskpath, picked_task)
# print(mask_sample.shape)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # load pretrained params
    if run_mode in ['test', 'finetune']:
        ckpt = tf.train.get_checkpoint_state(pretrain_model_path)
        if ckpt:
            ckpt_states = ckpt.all_model_checkpoint_paths
            saver.restore(sess, ckpt_states[pretrain_model_idx])
            logger.info('===> Load pretrained model from: ' +
                        pretrain_model_path)
        else:
            logger.error('===> No pretrained model found')
            raise FileNotFoundError('No pretrained model found')

    # [==> train & finetune]
    for task_index in range(num_task):
        saver = tf.train.Saver()
        logger.info('\n===== Adaptation for Task Task {:4d}/{:<4d} =====\n'.format(task_index,len(picked_task)))
        mask_sample_i = mask_sample[task_index]
        mask_s_sample_i = mask_s_sample[task_index]
        for epoch in range(Epoch):
            random.shuffle(nameList)
            epoch_loss = 0
            begin = time.time()
            
            max_iter = len(nameList) if len(nameList)<max_iter else max_iter # max iter in an epoch
            for iter in tqdm(range(int(max_iter/batch_size))):
                sample_name = nameList[iter *batch_size: (iter+1)*batch_size]
                gt_sample = np.zeros([batch_size, image_dim, image_dim, num_frame])
                meas_sample = np.zeros([batch_size, image_dim, image_dim])
                
                for index in range(len(sample_name)):
                    
                    gt_tmp = sci.loadmat(datadir + sample_name[index])
                    if "patch_save" in gt_tmp:
                        gt_tmp = gt_tmp['patch_save'] / 255
                    elif "orig" in gt_tmp:
                        gt_tmp = gt_tmp['orig'] / 255

                    meas_tmp, gt_tmp = generate_meas(gt_tmp, mask_sample_i)  # zzh: calculate meas


                    gt_sample[index,:, :] = gt_tmp[0, ...]
                    meas_sample[index,:, :] = meas_tmp[0, ...]

                meas_re_sample = meas_sample / mask_s_sample_i
                meas_re_sample = np.expand_dims(meas_re_sample, axis=-1)

                _, Loss = sess.run([optimizer, output['loss']],
                                feed_dict={mask: mask_sample_i,
                                            meas_re: meas_re_sample,
                                            gt: gt_sample})
                epoch_loss += Loss

            end = time.time()
            logger.info("===> Epoch {} Complete: Avg. Loss: {:.7f} \t Time: {:.2f}".format(
                epoch, epoch_loss / int(len(nameList)/batch_size), (end - begin)))

            if (epoch+1) % step == 0:
                # save model
                mode_save_path = save_path + 'trained_model/adapt_model%d/'%picked_task[task_index]
                if not ope(mode_save_path):
                    os.makedirs(mode_save_path)
                saver.save(sess, mode_save_path + model_name_prefix+str(picked_task[task_index]) + '.ckpt',
                        global_step=epoch, write_meta_graph=False)
                logger.info('---> adapt model #{} saved to: '.format(picked_task[task_index]) + mode_save_path)

                # eval & save recon (one coded meas)
                validset_psnr = 0
                validset_ssim = 0
            
                psnr_all = np.zeros(num_frame)
                ssim_all = np.zeros(num_frame)
               
                # make sure only weights_m updated
                # print('\n*****************\n', sess.run(weights['w2']))
                # print('\n*****************\n', sess.run(weights_m['w7_L']))
                
                for index in range(len(valid_nameList)):
                    # load data
                    data_tmp = sci.loadmat(
                        valid_dir + valid_nameList[index])

                    if "patch_save" in data_tmp:
                        gt_sample = data_tmp['patch_save'] / 255
                    elif "orig" in data_tmp:
                        gt_sample = data_tmp['orig'] / 255
                    else:
                        raise FileNotFoundError('No ORIG in dataset')
                    meas_sample, gt_sample = generate_meas(gt_sample, mask_sample_i)

                    # normalize data
                    mask_max = np.max(mask_sample_i)
                    mask_sample_i = mask_sample_i/mask_max
                    mask_s_sample_i = mask_s_sample_i/mask_max
                    meas_sample = meas_sample/mask_max
                    
                    meas_sample_re = meas_sample / mask_s_sample_i
                    meas_sample_re = np.expand_dims(meas_sample_re, -1)

                    # test data
                    pred = sess.run([output['pred']],
                                    feed_dict={mask: mask_sample_i,
                                            meas_re: meas_sample_re,
                                            gt: gt_sample})  # pred for Y_meas

                    pred = np.array(pred[0])
                    pred = np.squeeze(pred)
                    gt_sample = np.squeeze(gt_sample)

                    # eval: psnr, ssim

                    for k in range(num_frame):
                        psnr_all[k], ssim_all[k] = cal_psnrssim(
                            gt_sample[..., k], pred[..., k])

                    mean_psnr = np.mean(psnr_all)
                    mean_ssim = np.mean(ssim_all)

                    validset_psnr += mean_psnr
                    validset_ssim += mean_ssim

                    # save 1st data's recon image and data as an example
                    if index == 3:
                        plot_multi(pred, 'MeasRecon_Task%d_%s_Epoch%d' % (picked_task[task_index], valid_nameList[index].split('.')[0], epoch), col_num=num_frame//2, titles=psnr_all, savename='MeasRecon_Task%d_%s_Epoch%d_psnr%.2f_ssim%.2f' % (
                            picked_task[task_index], valid_nameList[index].split('.')[0], epoch, mean_psnr, mean_ssim), savedir=save_path+'recon_img/adapt_model%d/'%picked_task[task_index])

                validset_psnr = validset_psnr/len(valid_nameList)
                validset_ssim = validset_ssim/len(valid_nameList)
                logger.info('---> Aver. PSNR {:.2f}, Aver.SSIM {:.2f}'.format(validset_psnr, validset_ssim))
