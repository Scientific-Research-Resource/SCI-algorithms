# training model without attention or adversial training
from dataLoadess import OrigTrainDataset,OrigRandomMaskTrainDataset
from torch.utils.data import DataLoader
# from models import forward_rnn, cnn1, backrnn             # with attention
from models_wo_atten import forward_rnn, cnn1, backrnn      # without attention
from utils import generate_masks,generate_random_masks, time2file_name
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import logging
import numpy as np
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim


### environ
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

### setting
## path
train_data_path = "/data/zzh/dataset/SCI/training_truth/data_augment_256_10f"  # traning data from DAVIS2017
# train_data_path = '/data/zzh/project/RNN_SCI/Data/data_simu/testing_truth/bm_256_10f/' # for test
mask_path = "/data/zzh/project/MD_SCI/BIRNAT/test/mask/mask"
test_path = '/data/zzh/dataset/SCI/benchmark/benchmark_simu_ly_256'   # simulation benchmark data for comparison
# test_path = "/data/zzh/project/RNN_SCI/Data/data_simu/testing_truth/exp_256"  # experiment simulation benchmark data for comparison


## param
pretrained_model = 'binary_mask_256_Cr8_official'
mask_name = 'binary_mask_256_8f.mat'
Cr =8
# mask_size = (256,256,Cr)
block_size = 256
last_train = 100
max_iter = 100
batch_size = 1
learning_rate = 0.0003
lr_decay = 0.95
lr_decay_step = 3   # epoch interval for learning rate decay
checkpoint_step = 1 # epoch interval for save checkpoints


## data set
mask, mask_s = generate_masks(mask_path, mask_name)
# mask, mask_s = generate_random_masks(mask_size)
# dataset = OrigTrainDataset(train_data_path)
dataset = OrigRandomMaskTrainDataset(train_data_path)

train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


## model set
first_frame_net = cnn1(Cr+1).cuda()
rnn1 = forward_rnn().cuda()
rnn2 = backrnn().cuda()


if last_train != 0:
    first_frame_net = torch.load(
        './model/' + pretrained_model + "/first_frame_net_model_epoch_{}.pth".format(last_train))
    rnn1 = torch.load('./model/' + pretrained_model + "/rnn1_model_epoch_{}.pth".format(last_train))
    rnn2 = torch.load('./model/' + pretrained_model + "/rnn2_model_epoch_{}.pth".format(last_train))
    print('pre-trained model: \'{} - No. {} epoch\' loaded!'.format(pretrained_model, last_train))

loss = nn.MSELoss()
loss.cuda()



### function
## test
def test(test_path, epoch, result_path, logger):
    test_list = os.listdir(test_path)
    psnr_forward = torch.zeros(len(test_list))
    psnr_backward = torch.zeros(len(test_list))
    ssim_forward = torch.zeros(len(test_list))
    ssim_backward = torch.zeros(len(test_list))

    # load test data
    for i in range(len(test_list)):
        # load orig pic
        pic = scio.loadmat(test_path + '/' + test_list[i])

        if "orig" in pic:
            pic = pic['orig']
            sign = 1
        elif "patch_save" in pic:
            pic = pic['patch_save']
            sign = 0
        pic = pic / 255

        # calc meas
        pic_gt = np.zeros([pic.shape[2] // Cr, Cr, block_size, block_size])
        for jj in range(pic.shape[2] // Cr*Cr):
            if jj % Cr == 0:
                meas_t = np.zeros([block_size, block_size])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // Cr, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == Cr-1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % Cr == 0: #zzh
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        
        # calc
        meas = torch.from_numpy(meas)
        pic_gt = torch.from_numpy(pic_gt)
        meas = meas.cuda()
        pic_gt = pic_gt.cuda()
        meas = meas.float()
        pic_gt = pic_gt.float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)
        
        with torch.no_grad():
            h0 = torch.zeros(meas.shape[0], 20, block_size, block_size).cuda()
            xt1 = first_frame_net(mask, meas_re, block_size, Cr)
            out_pic1,h1 = rnn1(xt1, meas, mask, h0, meas_re, block_size, Cr)
            out_pic2 = rnn2(out_pic1, meas, mask, h1, meas_re, block_size, Cr)        #  out_pic1[:, fn-1, :, :]
        
        # calculate psnr and ssim
            psnr_1 = 0
            psnr_2 = 0
            ssim_1 = 0
            ssim_2 = 0

            for ii in range(meas.shape[0] * Cr):
                out_pic_forward = out_pic1[ii // Cr, ii % Cr, :, :]
                out_pic_backward = out_pic2[ii // Cr, ii % Cr, :, :]
                gt_t = pic_gt[ii // Cr, ii % Cr, :, :]
                mse_forward = loss(out_pic_forward * 255, gt_t * 255)
                mse_forward = mse_forward.data
                mse_backward = loss(out_pic_backward * 255, gt_t * 255)
                mse_backward = mse_backward.data
                psnr_1 += 10 * torch.log10(255 * 255 / mse_forward)
                psnr_2 += 10 * torch.log10(255 * 255 / mse_backward)

                ssim_1 += ssim(out_pic_forward.cpu().numpy(), gt_t.cpu().numpy())
                ssim_2 += ssim(out_pic_backward.cpu().numpy(), gt_t.cpu().numpy())

            psnr_1 = psnr_1 / (meas.shape[0] * Cr)
            psnr_2 = psnr_2 / (meas.shape[0] * Cr)
            psnr_forward[i] = psnr_1
            psnr_backward[i] = psnr_2

            ssim_1 = ssim_1 / (meas.shape[0] * Cr)
            ssim_2 = ssim_2 / (meas.shape[0] * Cr)
            ssim_forward[i] = ssim_1
            ssim_backward[i] = ssim_2

            # test performance
            if sign == 1:
                if epoch % 5 == 0 or (epoch > 50 and epoch % 2 == 0):
                    a = test_list[i]
                    name1 = result_path + '/forward_' + a[0:len(a) - 4] + '{}_{:.4f}_{:.4f}'.format(epoch, psnr_1, ssim_1) + '.mat'
                    name2 = result_path + '/backward_' + a[0:len(a) - 4] + '{}_{:.4f}_{:.4f}'.format(epoch, psnr_2, ssim_2) + '.mat'
                    out_pic1 = out_pic1.cpu()
                    out_pic2 = out_pic2.cpu()
                    scio.savemat(name1, {'pic': out_pic1.numpy()})
                    scio.savemat(name2, {'pic': out_pic2.numpy()})
    logger.info("only forward rnn result (psnr/ssim): {:.4f}/{:.4f}   backward rnn result: {:.4f}/{:.4f}"\
        .format(torch.mean(psnr_forward), torch.mean(ssim_forward), torch.mean(psnr_backward), torch.mean(ssim_backward)))

## train
def train(epoch, learning_rate, result_path, logger):
    epoch_loss = 0
    optimizer_g = optim.Adam([{'params': first_frame_net.parameters()}, {'params': rnn1.parameters()},
                            {'params': rnn2.parameters()}], lr=learning_rate)

    
    # if __name__ == '__main__':
    for iteration, batch in enumerate(train_data_loader):
        gt, meas = Variable(batch[0]), Variable(batch[1])
        gt = gt.cuda()  # [batch,Cr,block_size,block_size]
        gt = gt.float()
        meas = meas.cuda()  # [batch,block_size block_size]
        meas = meas.float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        

        batch_size1 = gt.shape[0]
        # print(meas.shape,gt.shape) #zzh debug
        # Cr = gt.shape[1]
        
        h0 = torch.zeros(batch_size1, 20, block_size, block_size).cuda()
        xt1 = first_frame_net(mask, meas_re, block_size, Cr)
        model_out1, h1 = rnn1(xt1, meas, mask, h0, meas_re, block_size, Cr)
        model_out = rnn2(model_out1, meas, mask, h1,meas_re, block_size, Cr)           #  model_out1[:, fn-1, :, :]
        
        optimizer_g.zero_grad()
        Loss1 = loss(model_out1, gt)
        Loss2 = loss(model_out, gt)
        Loss = 0.5 * Loss1 + 0.5 * Loss2

        epoch_loss += Loss.data

        Loss.backward()
        optimizer_g.step()

        # show loss and time
        if iteration%50==0:
            now_time = time.time()
            print('---> iter {} Complete: Avg. Loss: {:.8f} time: {:.2f}'\
                .format(iteration, epoch_loss / iteration, now_time - begin))
            
    test(test_path, epoch, result_path, logger)

    end = time.time()
    logger.info('===> Epoch {} Complete: Avg. Loss: {:.8f} time: {:.2f}'\
        .format(epoch, epoch_loss / len(train_data_loader), end - begin))

## checkpoint
def checkpoint(epoch, model_path, logger):
    model_out_path = './' + model_path + '/' + "first_frame_net_model_epoch_{}.pth".format(epoch)
    torch.save(first_frame_net, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def checkpoint2(epoch, model_path):
    model_out_path = './' + model_path + '/' + "rnn1_model_epoch_{}.pth".format(epoch)
    torch.save(rnn1, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def checkpoint3(epoch, model_path):
    model_out_path = './' + model_path + '/' + "rnn2_model_epoch_{}.pth".format(epoch)
    torch.save(rnn2, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def main(learning_rate):
    # prepare
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # train
    print('\n---- start training ----\n')
    logger.info('Code: train_random_mask.py') 
    logger.info('train mask: random; test mask: {}'.format(mask_path + '/' + mask_name))
    if last_train != 0:
        logger.info('loading pre-trained model: \'{} - No. {} epoch\'...'.format(pretrained_model, last_train))

    for epoch in range(last_train + 1, last_train + max_iter + 1):
        train(epoch, learning_rate, result_path, logger)
        if (epoch % checkpoint_step == 0 or epoch > 70):
            checkpoint(epoch, model_path, logger)
            checkpoint2(epoch, model_path)
            checkpoint3(epoch, model_path)
        if (epoch % lr_decay_step == 0) and (epoch < 150):
            learning_rate = learning_rate * lr_decay
            logger.info('current learning rate: {}\n'.format(learning_rate))


if __name__ == '__main__':
    begin = time.time()
    main(learning_rate)
