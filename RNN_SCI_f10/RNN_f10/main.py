"""
@author : Ziheng Cheng, Bo Chen
@Email : zhcheng@stu.xidian.edu.cn      bchen@mail.xidian.edu.cn

Description:
    This is the train code for Snapshot Compressive Imaging reconstruction in recurrent convolution neural network

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

from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import forward_rnn, cnn1, backrnn
from utils import generate_masks, time2file_name
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

#data_path = "../Data/B_8_DAVIS2017/train/480p"  # traning data from DAVIS2017
data_path = "../Data/B_8_DAVIS2017/train/480p"  # traning data from DAVIS2017

test_path1 = "../Data/test/simulation"  # simulation data for comparison

mask_name = 'combine_binary_mask_256_10f.mat';

mask, mask_s = generate_masks(data_path, mask_name)

last_train = 0
model_save_filename = ''
max_iter = 100
batch_size = 3
learning_rate = 0.0003
mode = 'train'  # train or test

dataset = Imgdataset(data_path)

train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

first_frame_net = cnn1().cuda()
rnn1 = forward_rnn().cuda()
rnn2 = backrnn().cuda()

if last_train != 0:
    first_frame_net = torch.load(
        './model/' + model_save_filename + "/first_frame_net_model_epoch_{}.pth".format(last_train))
    rnn1 = torch.load('./model/' + model_save_filename + "/rnn1_model_epoch_{}.pth".format(last_train))
    rnn2 = torch.load('./model/' + model_save_filename + "/rnn2_model_epoch_{}.pth".format(last_train))

loss = nn.MSELoss()
loss.cuda()


def test(test_path, epoch, result_path):
    test_list = os.listdir(test_path)
    psnr_forward = torch.zeros(len(test_list))
    psnr_backward = torch.zeros(len(test_list))
    ## load test data
    '''
    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])

        if "orig" in pic:
            pic = pic['orig']
            sign = 1
        elif "patch_save" in pic:
            pic = pic['patch_save']
            sign = 0
        elif "p1" in pic:
            pic = pic['p1']
            sign = 0
        elif "p2" in pic:
            pic = pic['p2']
            sign = 0
        elif "p3" in pic:
            pic = pic['p3']
            sign = 0
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // 8, 8, 256, 256])
        for jj in range(pic.shape[2]):
            if jj % 8 == 0:
                meas_t = np.zeros([256, 256])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // 8, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % 8 == 0 and jj != 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        '''
        meas = torch.from_numpy(meas)
        pic_gt = torch.from_numpy(pic_gt)
        meas = meas.cuda()
        pic_gt = pic_gt.cuda()
        meas = meas.float()
        pic_gt = pic_gt.float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)
        
        with torch.no_grad():
            h0 = torch.zeros(meas.shape[0], 20, 256, 256).cuda()
            xt1 = first_frame_net(meas, mask, meas.shape[0], meas_re)
            out_pic1,h1 = rnn1(xt1, meas, mask, meas.shape[0], h0, mode, meas_re)
            out_pic2 = rnn2(out_pic1[:, 9, :, :], meas, mask, meas.shape[0], h1, mode, meas_re)        #  out_pic1[:, fn-1, :, :]
        # calculate psnr
        '''
            psnr_1 = 0
            psnr_2 = 0
            for ii in range(meas.shape[0] * 8):
                out_pic_forward = out_pic1[ii // 8, ii % 8, :, :]
                out_pic_backward = out_pic2[ii // 8, ii % 8, :, :]
                gt_t = pic_gt[ii // 8, ii % 8, :, :]
                mse_forward = loss(out_pic_forward * 255, gt_t * 255)
                mse_forward = mse_forward.data
                mse_backward = loss(out_pic_backward * 255, gt_t * 255)
                mse_backward = mse_backward.data
                psnr_1 += 10 * torch.log10(255 * 255 / mse_forward)
                psnr_2 += 10 * torch.log10(255 * 255 / mse_backward)
            psnr_1 = psnr_1 / (meas.shape[0] * 8)
            psnr_2 = psnr_2 / (meas.shape[0] * 8)
            psnr_forward[i] = psnr_1
            psnr_backward[i] = psnr_2

            if sign == 1:
                if epoch % 10 == 0 or (epoch > 50 and epoch % 2 == 0):
                    a = test_list[i]
                    name1 = result_path + '/forward_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_1) + '.mat'
                    name2 = result_path + '/backward_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_2) + '.mat'
                    out_pic1 = out_pic1.cpu()
                    out_pic2 = out_pic2.cpu()
                    scio.savemat(name1, {'pic': out_pic1.numpy()})
                    scio.savemat(name2, {'pic': out_pic2.numpy()})
    print("only forward rnn result: {:.4f}".format(torch.mean(psnr_forward)),
          "     backward rnn result: {:.4f}".format(torch.mean(psnr_backward)))
          '''


def train(epoch, learning_rate, result_path):
    epoch_loss = 0
    begin = time.time()

    optimizer = optim.Adam([{'params': first_frame_net.parameters()}, {'params': rnn1.parameters()},
                            {'params': rnn2.parameters()}], lr=learning_rate)

    if __name__ == '__main__':
        for iteration, batch in enumerate(train_data_loader):
            gt, meas = Variable(batch[0]), Variable(batch[1])
            gt = gt.cuda()  # [batch,8,256,256]
            gt = gt.float()
            meas = meas.cuda()  # [batch,256 256]
            meas = meas.float()

            meas_re = torch.div(meas, mask_s)
            meas_re = torch.unsqueeze(meas_re, 1)

            optimizer.zero_grad()

            batch_size1 = gt.shape[0]

            h0 = torch.zeros(batch_size1, 20, 256, 256).cuda()

            xt1 = first_frame_net(meas, mask, batch_size1, meas_re)
            model_out1, h1 = rnn1(xt1, meas, mask, batch_size1, h0, mode, meas_re)
            model_out = rnn2(model_out1[:, 9, :, :], meas, mask, batch_size1, h1, mode, meas_re)           #  model_out1[:, fn-1, :, :]

            Loss1 = loss(model_out1, gt)
            Loss2 = loss(model_out, gt)
            Loss = 0.5 * Loss1 + 0.5 * Loss2

            epoch_loss += Loss.data

            Loss.backward()
            optimizer.step()

        test(test_path1, epoch, result_path)

    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))


def checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "first_frame_net_model_epoch_{}.pth".format(epoch)
    torch.save(first_frame_net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def checkpoint2(epoch, model_path):
    model_out_path = './' + model_path + '/' + "rnn1_model_epoch_{}.pth".format(epoch)
    torch.save(rnn1, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def checkpoint3(epoch, model_path):
    model_out_path = './' + model_path + '/' + "rnn2_model_epoch_{}.pth".format(epoch)
    torch.save(rnn2, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def main(learning_rate):
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for epoch in range(last_train + 1, last_train + max_iter + 1):
        train(epoch, learning_rate, result_path)
        if (epoch % 10 == 0 or epoch > 70):
            checkpoint(epoch, model_path)
            checkpoint2(epoch, model_path)
            checkpoint3(epoch, model_path)
        if (epoch % 5 == 0) and (epoch < 150):
            learning_rate = learning_rate * 0.95
            print(learning_rate)


if __name__ == '__main__':
    main(learning_rate)
