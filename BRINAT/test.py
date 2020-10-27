from models import forward_rnn, cnn1, backrnn 
from utils import generate_masks, time2file_name
import torch.nn as nn
import torch
import scipy.io as scio
import datetime
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim


### environ
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

### setting
## path
#train_data_path = "/data/zzh/project/E2E_CNN/data_simu/training_truth/data_augment_256_10f"  # traning data from DAVIS2017
mask_path = "/data/zzh/project/RNN_SCI/Data/data_simu/exp_mask"
test_path = '/data/zzh/project/RNN_SCI/Data/data_simu/testing_truth/bm_256_10f/'   # simulation benchmark data for comparison
# test_path = '/data/zzh/project/RNN_SCI/Data/data_simu/testing_truth/exp_256'
train_data_path =test_path # for test

## param
pretrained_model = '2020_10_27_17_59_23'
mask_name = 'multiplex_shift_binary_mask_256_10f.mat'
Cr = 10
block_size = 256
last_train = 10


## data set
mask, mask_s = generate_masks(mask_path, mask_name)


## model set
first_frame_net = cnn1(Cr+1).cuda()
rnn1 = forward_rnn().cuda()
rnn2 = backrnn().cuda()

if last_train != 0:
    first_frame_net = torch.load(
        './model/' + pretrained_model + "/first_frame_net_model_epoch_{}.pth".format(last_train))
    rnn1 = torch.load('./model/' + pretrained_model + "/rnn1_model_epoch_{}.pth".format(last_train))
    rnn2 = torch.load('./model/' + pretrained_model + "/rnn2_model_epoch_{}.pth".format(last_train))

loss = nn.MSELoss()
loss.cuda()


## function
def test(test_path, epoch, result_path):
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
            sign = 0 #zzh?
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

            if sign == 1:
                # if epoch % 5 == 0 or (epoch > 50 and epoch % 2 == 0):
                a = test_list[i]
                name1 = result_path + '/forward_' + a[0:len(a) - 4] + '{}_{:.4f}_{:.4f}'.format(epoch, psnr_1, ssim_1) + '.mat'
                name2 = result_path + '/backward_' + a[0:len(a) - 4] + '{}_{:.4f}_{:.4f}'.format(epoch, psnr_2, ssim_2) + '.mat'
                out_pic1 = out_pic1.cpu()
                out_pic2 = out_pic2.cpu()
                scio.savemat(name1, {'pic': out_pic1.numpy()})
                scio.savemat(name2, {'pic': out_pic2.numpy()})
    print("only forward rnn result (psnr/ssim): {:.4f}/{:.4f}   backward rnn result: {:.4f}/{:.4f}"\
        .format(torch.mean(psnr_forward), torch.mean(ssim_forward), torch.mean(psnr_backward), torch.mean(ssim_backward)))




def main():    
    result_path = 'recon' + '/' + pretrained_model
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    print('\n---- start testing ----\n')
    test(test_path, last_train, result_path)    
   

if __name__ == '__main__':
    main()
