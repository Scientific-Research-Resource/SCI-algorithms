"""
@author : Hao
# Base model training for metaSCI
# modified: Zhihong Zhang, 2021.6

Note:

Todo:
+ realize evalution on eval dataset

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
from MetaFunc import construct_weights_modulation, MAML_modulation, forward_modulation

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
Total_batch_size = batch_size*2  # X_meas & Y_meas
num_frame = 10
image_dim = 256
Epoch = 100
sigmaInit = 0.01
init_lr = 5e-4
decay_rate = 0.9
decay_steps = 10000
step = 1
update_lr = 1e-5 # learning rate for self-designed multi-task gradient update
num_updates = 5  # num of update for self-designed multi-task gradient update
max_iter = 30000 # max iter in one epoch
picked_task = [200]  # pick masks for base model train
num_task = len(picked_task)  # num of picked masks
run_mode = 'finetune'  # 'train', 'test','finetune'
test_real = False  # test real data
pretrain_model_idx = -1  # pretrained model index, 0 for no pretrained
# exp_name = "Realmask_BaseTrain_256_Cr10_zzhTest"
exp_name = "simulate_data_256_Cr10_zzhTest"
timestamp = '{:%m-%d_%H-%M}'.format(datetime.now())  # date info

# data path
# datadir = "../[data]/dataset/training_truth/data_augment_256_8f_demo/"
# maskpath = "./dataset/mask/origDemo_mask_256_Cr8_4.mat"
# trainning set
datadir = "../[data]/dataset/training_truth/data_augment_256_10f_demo/"
valid_dir = "../[data]/dataset/testing_truth/test_256_10f/"
maskpath = "./dataset/mask/realMask_256_Cr10_N576_overlap50.mat"
# datadir = "../[data]/dataset/training_truth/data_augment_512_10f/"
# maskpath = "./dataset/mask/demo_mask_512_Cr10_N4.mat"

# model path
# pretrain_model_path = './result/_pretrained_model/simulate_data_256_Cr8/'
pretrain_model_path = './result/train/M_Realmask_data_256_Cr10_zzhTest_06-17_21-10/trained_model/'

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
logger.info('\t Params: batch_size {:d}, num_frame {:d}, image_dim {:d}, sigmaInit {:f}, init_lr {:f}, decay_rate {:f}, decay_steps {:d}, update_lr {:f}, num_updates {:d}, max_iter {:d}, picked_task {:s}, run_mode- {:s}, pretrain_model_idx {:d}'.format(
    batch_size, num_frame, image_dim, sigmaInit, init_lr, decay_rate, decay_steps, update_lr, num_updates, max_iter, str(picked_task), run_mode, pretrain_model_idx))

# %% construct graph, load pretrained params ==> train, finetune, test
weights, weights_m = construct_weights_modulation(sigmaInit, num_frame)

# For train
mask = tf.placeholder('float32', [num_task, image_dim, image_dim, num_frame])
X_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
X_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])
Y_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
Y_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])

final_output = MAML_modulation(mask, X_meas_re, X_gt, Y_meas_re, Y_gt, weights,
                               weights_m, batch_size, num_frame, image_dim, update_lr, num_updates)

# learning rate
global_step = tf.Variable(tf.constant(0), trainable=False)
learning_rate = tf.train.exponential_decay(init_lr,
                                           global_step=global_step,
                                           decay_steps=decay_steps,
                                           decay_rate=decay_rate)
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(final_output['loss'],global_step=global_step)
saver = tf.train.Saver()

# For eval
eval_mask = tf.placeholder('float32', [image_dim, image_dim, num_frame])
eval_meas_re = tf.placeholder('float32', [batch_size, image_dim, image_dim, 1])
eval_gt = tf.placeholder(
    'float32', [batch_size, image_dim, image_dim, num_frame])
pred_output = forward_modulation(
    eval_mask, eval_meas_re, eval_gt, weights, weights_m, batch_size, num_frame, image_dim)

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
            logger.warn('===> No pretrained model found, skip')

    # [==> train & finetune]
    for epoch in range(Epoch):
        random.shuffle(nameList)
        epoch_loss = 0
        begin = time.time()

        max_iter = len(nameList) if len(nameList)<max_iter else max_iter # max iter in an epoch
        for iter in tqdm(range(int(max_iter/Total_batch_size))):
            sample_name = nameList[iter *
                                    Total_batch_size: (iter+1)*Total_batch_size]
            X_gt_sample = np.zeros([num_task, batch_size, image_dim, image_dim, num_frame])
            X_meas_sample = np.zeros([num_task, batch_size, image_dim, image_dim])
            Y_gt_sample = np.zeros([num_task, batch_size, image_dim, image_dim, num_frame])
            Y_meas_sample = np.zeros([num_task, batch_size, image_dim, image_dim])

            for task_index in range(num_task):
                mask_sample_i = mask_sample[task_index]
                for index in range(len(sample_name)):
                    gt_tmp = sci.loadmat(datadir + sample_name[index])
                    if "patch_save" in gt_tmp:
                        gt_tmp = gt_tmp['patch_save'] / 255
                    elif "orig" in gt_tmp:
                        gt_tmp = gt_tmp['orig'] / 255

                    meas_tmp, gt_tmp = generate_meas(gt_tmp, mask_sample_i)  # zzh: calculate meas

                    if index < batch_size:
                        X_gt_sample[task_index, index,
                                    :, :] = gt_tmp[0, ...]
                        X_meas_sample[task_index, index,
                                        :, :] = meas_tmp[0, ...]
                    else:
                        Y_gt_sample[task_index, index -
                                    batch_size, :, :] = gt_tmp
                        Y_meas_sample[task_index, index -
                                        batch_size, :, :] = meas_tmp

            X_meas_re_sample = X_meas_sample / np.expand_dims(mask_s_sample, axis=1)
            X_meas_re_sample = np.expand_dims(X_meas_re_sample, axis=-1)

            Y_meas_re_sample = Y_meas_sample / np.expand_dims(mask_s_sample, axis=1)
            Y_meas_re_sample = np.expand_dims(Y_meas_re_sample, axis=-1)

            _, Loss = sess.run([optimizer, final_output['loss']],
                                feed_dict={mask: mask_sample,
                                            X_meas_re: X_meas_re_sample,
                                            X_gt: X_gt_sample,
                                            Y_meas_re: Y_meas_re_sample,
                                            Y_gt: Y_gt_sample})
            
            # debug: learning rate
            # rate = sess.run([learning_rate])
            # print('curent learning rate:', rate)  
                    
            epoch_loss += Loss

        end = time.time()
        logger.info("===> Epoch {} Complete: Avg. Loss: {:.7f} \t Time: {:.2f}".format(
            epoch, epoch_loss / int(len(nameList)/batch_size), (end - begin)))

        if (epoch+1) % step == 0:
            # save model
            saver.save(sess, save_path + 'trained_model/model.ckpt',
                        global_step=epoch, write_meta_graph=False)
            logger.info('---> model saved to: ' + save_path)

            # eval & save recon (one coded meas)
            validset_psnr = 0
            validset_ssim = 0
            for task_index in range(num_task):
                psnr_all = np.zeros(num_frame)
                ssim_all = np.zeros(num_frame)
                mask_sample_i = mask_sample[task_index]
                mask_s_sample_i = mask_s_sample[task_index]

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
                    pred = sess.run([pred_output['pred']],
                                    feed_dict={eval_mask: mask_sample_i,
                                                eval_meas_re: meas_sample_re,
                                                eval_gt: gt_sample})  # pred for Y_meas

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

                    # save 1st data's recon image and data
                    if index == 3:
                        plot_multi(pred, 'MeasRecon_Task%d_%s_Epoch%d' % (task_index, valid_nameList[index].split('.')[0], epoch), col_num=num_frame//2, titles=psnr_all, savename='MeasRecon_Task%d_%s_Epoch%d_psnr%.2f_ssim%.2f' % (
                            task_index, valid_nameList[index].split('.')[0], epoch, mean_psnr, mean_ssim), savedir=save_path+'recon_img/')

                validset_psnr = validset_psnr/len(valid_nameList)
                validset_ssim = validset_ssim/len(valid_nameList)
                lr_now = sess.run([learning_rate])
                logger.info('---> Aver. PSNR {:.2f}, Aver.SSIM {:.2f} Learning rate {:.5e}'.format(validset_psnr, validset_ssim, lr_now[0]))
