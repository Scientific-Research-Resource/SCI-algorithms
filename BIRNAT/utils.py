import torch
import scipy.io as scio
import numpy as np
import cv2
import scipy.io as scio
from os.path import join as opj

def generate_masks(mask_path, mask_name = 'mask.mat'): # zzh
    mask = scio.loadmat(mask_path + '/' + mask_name)
    mask = mask['mask']
    mask = np.transpose(mask, [2, 0, 1])

    mask_s = np.sum(mask, axis=0)

    # replace 0 to avoid nan value
    if (mask - mask.astype(np.int32)).any():
        # for float mask
        index = np.where(mask_s == 0)
        mask_s[index] = 0.1 # zzh: this value is chosen empirically
        # print('float mask') #[debug]
    else:
        # for binary mask
        index = np.where(mask_s == 0)
        mask_s[index] = 1
        mask_s = mask_s.astype(np.uint8) 
        # print('binary mask') #[debug]
        
    # print('\nmask: {}'.format(mask_path + '/' + mask_name)) #[debug]

    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s

def generate_random_masks(mask_size): # zzh
    # mask = scio.loadmat(mask_path + '/' + mask_name)
    # mask = mask['mask']
    # mask = np.transpose(mask, [2, 0, 1])
    mask = np.random.randint(0,2, size=mask_size).astype(float)

    mask_s = np.sum(mask, axis=0)

    # replace 0 to avoid nan value
    if (mask - mask.astype(np.int32)).any():
        # for float mask
        index = np.where(mask_s == 0)
        mask_s[index] = 0.1 # zzh: this value is chosen empirically
        # print('float mask') #[debug]
    else:
        # for binary mask
        index = np.where(mask_s == 0)
        mask_s[index] = 1
        mask_s = mask_s.astype(np.uint8) 
        # print('binary mask') #[debug]
        
    # print('\nmask: {}'.format(mask_path + '/' + mask_name)) #[debug]

    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def save_test_result(result_path,test_datname,epoch,out_pic2,pic_gt,psnr_2, ssim_2,block_size):
    # save test result (only the backward result is saved, the forward result is commented)
    # 'out_pic2','pic_gt' are tensor

    # name1 = result_path + '/forward_' + test_datname[0:len(test_datname) - 4] + '{}_{:.4f}_{:.4f}'.format(epoch, psnr_1, ssim_1) + '.mat'
    name2 = result_path + '/backward_' + test_datname[0:len(test_datname) - 4] + '{}_{:.4f}_{:.4f}'.format(epoch, psnr_2, ssim_2) + '.mat'
    # scio.savemat(name1, {'pic': out_pic1.cpu().numpy()})
    scio.savemat(name2, {'pic': out_pic2.cpu().numpy()})
    
    meas_num,Cr = out_pic2.shape[0:2]
    
    # save gt v.s. recon (mid frame of mid measurement)
    gtVSrec = np.concatenate((out_pic2.cpu().numpy()[meas_num//2,Cr//2,:,:]*255.0,
                    pic_gt.cpu().numpy()[meas_num//2,Cr//2,:,:]*255.0),1)
    cmp_png_name = opj(result_path, 'backward_' + test_datname[0:len(test_datname) - 4] + '_cmp_meas{:02d}_idx{:02d}_{:02d}_{:.4f}'.format(meas_num//2,Cr//2, epoch, psnr_2) + '.png')                            
    cv2.imwrite(cmp_png_name, gtVSrec, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # save all recon
    row_num = 2
    col_num = np.ceil(Cr/row_num).astype(int)
    rec_all = np.zeros((block_size*row_num, block_size*col_num))
    
    for i in range(meas_num):
        rec_frames = out_pic2[i,:,:,:].cpu().numpy()
        for j in range(Cr):
            rec_all[(j//col_num) *block_size:(j//col_num+1) *block_size, (j%col_num) *block_size:(j%col_num+1) *block_size] = rec_frames[j,:,:] 
        recon_png_name = opj(result_path, 'backward_' + test_datname[0:len(test_datname) - 4] + '_recon_meas{:02d}_{:02d}_{:.4f}'.format(i, epoch, psnr_2) + '.png')      
        cv2.imwrite(recon_png_name , rec_all*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])     