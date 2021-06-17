from utils import generate_masks, time2file_name
import torch.nn as nn
import torch
import scipy.io as scio
import datetime
import os
import numpy as np
import argparse
from utils import compare_ssim, compare_psnr

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "./train"
test_path1 = "./test"

parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--last_train', default=100, type=int, help='pretrain model')
parser.add_argument('--model_save_filename', default='./model/', type=str, help='pretrain model save folder name')
parser.add_argument('--max_iter', default=100, type=int, help='max epoch')
parser.add_argument('--learning_rate', default=0.0002, type=float)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--B', default=8, type=int, help='compressive rate')
parser.add_argument('--num_block', default=18, type=int, help='the number of reversible blocks')
parser.add_argument('--num_group', default=2, type=int, help='the number of groups')
parser.add_argument('--size', default=[256, 256], type=int, help='input image resolution')
parser.add_argument('--mode', default='reverse', type=str, help='training mode: reverse or normal')


args = parser.parse_args()
mask, mask_s = generate_masks(data_path)

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
                    psnr_1 += compare_psnr(gt_t.cpu().numpy() * 255, out_pic_CNN.cpu().numpy() * 255)
                    ssim_1 += compare_ssim(gt_t.cpu().numpy() * 255, out_pic_CNN.cpu().numpy() * 255)

            psnr_cnn[i] = psnr_1 / (meas.shape[0] * args.B)
            ssim_cnn[i] = ssim_1 / (meas.shape[0] * args.B)

            a = test_list[i]
            name1 = result_path + '/RevSCInet_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_cnn[i]) + '.mat'
            out_save1 = out_save1.cpu()
            scio.savemat(name1, {'pic': out_save1.numpy()})
    print("RevSCInet result: PSNR -- {:.4f}, SSIM -- {:.4f}".format(torch.mean(psnr_cnn), torch.mean(ssim_cnn)))


if __name__ == '__main__':

    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.last_train != 0:
        rev_net = torch.load(
            './model/' + args.model_save_filename + "/RevSCInet_model_epoch_{}.pth".format(args.last_train))
        rev_net = rev_net.module if hasattr(rev_net, "module") else rev_net
    test(test_path1, args.last_train, result_path, rev_net.eval(), args)
