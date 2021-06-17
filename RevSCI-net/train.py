from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import re_3dcnn
from utils import generate_masks, time2file_name
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
import random
from torch.autograd import Variable
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n_gpu = torch.cuda.device_count()
print('The number of GPU is {}'.format(n_gpu))

data_path = "./train"
test_path1 = "./test"

mask, mask_s = generate_masks(data_path)

parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--last_train', default=0, type=int, help='pretrain model')
parser.add_argument('--model_save_filename', default='', type=str, help='pretrain model save folder name')
parser.add_argument('--max_iter', default=100, type=int, help='max epoch')
parser.add_argument('--learning_rate', default=0.0002, type=float)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--B', default=8, type=int, help='compressive rate')
parser.add_argument('--num_block', default=18, type=int, help='the number of reversible blocks')
parser.add_argument('--num_group', default=2, type=int, help='the number of groups')
parser.add_argument('--size', default=[256, 256], type=int, help='input image resolution')
parser.add_argument('--mode', default='normal', type=str, help='training mode: reverse or normal')


args = parser.parse_args()

dataset = Imgdataset(data_path)

train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

loss = nn.MSELoss()
loss.cuda()


def test(test_path, epoch, result_path, model, args):
    test_list = os.listdir(test_path)
    psnr_cnn, ssim_cnn = torch.zeros(len(test_list)), torch.zeros(len(test_list))
    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])

        if "orig" in pic:
            pic = pic['orig']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // args.B, args.B, args.size[0], args.size[1]])
        for jj in range(pic.shape[2]):
            if jj % args.B == 0:
                meas_t = np.zeros([args.size[0], args.size[1]])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // args.B, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % args.B == 0 and jj != args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas).cuda().float()
        pic_gt = torch.from_numpy(pic_gt).cuda().float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        out_save1 = torch.zeros([meas.shape[0], args.B, args.size[0], args.size[1]]).cuda()
        with torch.no_grad():

            psnr_1, ssim_1 = 0, 0
            for ii in range(meas.shape[0]):
                out_pic1 = model(meas_re[ii:ii + 1, ::], args)
                out_pic1 = out_pic1[0, ::]
                out_save1[ii, :, :, :] = out_pic1[0, :, :, :]
                for jj in range(args.B):
                    out_pic_CNN = out_pic1[0, jj, :, :]
                    gt_t = pic_gt[ii, jj, :, :]
                    psnr_1 += compare_psnr(gt_t.cpu().numpy(), out_pic_CNN.cpu().numpy())
                    ssim_1 += compare_ssim(gt_t.cpu().numpy(), out_pic_CNN.cpu().numpy())

            psnr_cnn[i] = psnr_1 / (meas.shape[0] * args.B)
            ssim_cnn[i] = ssim_1 / (meas.shape[0] * args.B)

            a = test_list[i]
            name1 = result_path + '/RevSCInet_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_cnn[i]) + '.mat'
            out_save1 = out_save1.cpu()
            scio.savemat(name1, {'pic': out_save1.numpy()})
    print("RevSCInet result: PSNR -- {:.4f}, SSIM -- {:.4f}".format(torch.mean(psnr_cnn), torch.mean(ssim_cnn)))


def train(epoch, result_path, model, args):
    epoch_loss = 0
    begin = time.time()

    optimizer_g = optim.Adam([{'params': model.parameters()}], lr=args.learning_rate)

    for iteration, batch in tqdm(enumerate(train_data_loader)):
        gt = Variable(batch)
        gt = gt.cuda().float()  # [batch,8,256,256]

        maskt = mask.expand([gt.shape[0], args.B, args.size[0], args.size[1]])
        meas = torch.mul(maskt, gt)
        meas = torch.sum(meas, dim=1)

        meas = meas.cuda().float()  # [batch,256 256]

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        optimizer_g.zero_grad()

        if args.mode == 'normal':
            xt1 = model(meas_re, args)
            Loss1 = loss(torch.squeeze(xt1), gt)
            Loss1.backward()
            optimizer_g.step()
        elif args.mode == 'reverse':
            xt1, Loss1 = model.for_backward(mask, meas_re, gt, loss, optimizer_g, args)

        epoch_loss += Loss1.data

    model = model.module if hasattr(model, "module") else model
    test(test_path1, epoch, result_path, model.eval(), args)
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))


def checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "RevSCInet_model_epoch_{}.pth".format(epoch)
    torch.save(rev_net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main(model, args):
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for epoch in range(args.last_train + 1, args.last_train + args.max_iter + 1):
        train(epoch, result_path, model, args)
        if (epoch % 5 == 0) and (epoch < 150):
            args.learning_rate = args.learning_rate * 0.95
            print(args.learning_rate)
        if (epoch % 5 == 0 or epoch > 50):
            model = model.module if hasattr(model, "module") else model
            checkpoint(epoch, model_path)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)


if __name__ == '__main__':
    print(args.mode)
    print(args.learning_rate)

    rev_net = re_3dcnn(args).cuda()
    rev_net.mask = mask
    if n_gpu > 1:
        rev_net = torch.nn.DataParallel(rev_net)
    if args.last_train != 0:
        rev_net = torch.load(
            './model/' + args.model_save_filename + "/RevSCInet_model_epoch_{}.pth".format(args.last_train))
        rev_net = rev_net.module if hasattr(rev_net, "module") else rev_net
    main(rev_net, args)
