import os
import numpy as np
from utils import cli_run
# %%
# [0] pre-params setting
mask_name = 'multiplex_shift_binary_mask'
orig_name_all = ['football', 'hummingbird',
                 'ReadySteadyGo', 'Jockey', 'YachtRide']
scale_all = ['256', '512', '1024']
Cr_all = [10, 20]

test_algo_flag_all = ['all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet',
                      'gapfastdvdnet', 'admmfastdvdnet', 'gaptv+ffdnet', 'gaptv+fastdvdnet']

MAXB = 255
img_num=48

opti_tv_weight_table_exp1 = {'256_Cr10': 0.20,
                   '256_Cr20': 0.15,
                   '512_Cr10': 0.25,
                   '512_Cr20': 0.10,
                   '1024_Cr10': 0.10,
                   '1024_Cr20': 0.10}
# opti_tv_weight_table_exp2 = {'binary_mask': 0.20,
#                         'shift_binary_mask': 0.15,
#                         'multiplex_shift_binary_mask': 0.25
#                         }


# %%
# [1] params choose
result_path = '/results/tmp'


# orig_names = ['Jockey']
# orig_names = ['YachtRide']
orig_names = ['football']
# orig_names = orig_name_all
scales = ['256']
Crs = [10]

show_res_flag = 1
save_res_flag = 1
log_result_flag = 1
# test_algo_flag = 'gaptv'
test_algo_flag = 'gapffdnet'
# test_algo_flag = test_algo_flag_all[5]

iframe = 0                 # from which frame of meas to recon            
nframe = 1       # how many frame of meas to recon [img_num//Cr ]


sigma1 = 0     # scaler
iter_max1 = 0 


# %%
# [2] run
sigma2 = [50/255, 25/255, 12/255]
iter_max2 = [40, 50, 70]
print('\n\n 1----------------------\n\n')
for scale in scales:
    for Cr in Crs:
        for orig_name in orig_names:
            # result_path = '/results/exp1_multiscale/'+test_algo_flag+'/'+scale
            cli_run('pnp_sci_video_orig_simuexp1.py', orig_name, scale, Cr, mask_name, test_algo_flag,
            result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
            show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag,
            tv_weight = opti_tv_weight_table_exp1[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)



sigma2 = [50/255, 25/255, 12/255]
iter_max2 = [50, 40, 70]
print('\n\n 2----------------------\n\n')
for scale in scales:
    for Cr in Crs:
        for orig_name in orig_names:
            # result_path = '/results/exp1_multiscale/'+test_algo_flag+'/'+scale
            cli_run('pnp_sci_video_orig_simuexp1.py', orig_name, scale, Cr, mask_name, test_algo_flag,
            result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
            show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag,
            tv_weight = opti_tv_weight_table_exp1[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)
            
            

sigma2 = [50/255, 25/255, 12/255, 6/255]
iter_max2 = [30, 30, 50, 50]
print('\n\n 3----------------------\n\n')
for scale in scales:
    for Cr in Crs:
        for orig_name in orig_names:
            # result_path = '/results/exp1_multiscale/'+test_algo_flag+'/'+scale
            cli_run('pnp_sci_video_orig_simuexp1.py', orig_name, scale, Cr, mask_name, test_algo_flag,
            result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
            show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag,
            tv_weight = opti_tv_weight_table_exp1[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)