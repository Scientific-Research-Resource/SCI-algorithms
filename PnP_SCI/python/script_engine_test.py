import os
import numpy as np
from my_tools import cli_run
# %%
# [0] pre-params setting
orig_name_all = ['football', 'hummingbird',
                 'ReadySteadyGo', 'Jockey', 'YachtRide']
scale_all = ['256', '512', '1024']
Cr_all = [10, 20]

gaussian_noise_level_all = [10, 20, 40, 80]

test_algo_flag_all = ['all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet',
                      'gapfastdvdnet', 'admmfastdvdnet', 'gaptv+ffdnet', 'gaptv+fastdvdnet']

mask_name_all = ['binary_mask', 'shift_binary_mask', 'multiplex_shift_binary_mask2']

MAXB = 255
img_num=48
# opti_tv_weight_table_exp1 = {'256_Cr10': 0.20,
#                         '256_Cr20': 0.15,
#                         '512_Cr10': 0.25,
#                         '512_Cr20': 0.10,
#                         '1024_Cr10': 0.10,
#                         '1024_Cr20': 0.10}
opti_tv_weight_table_exp2 = {'binary_mask': 0.15,
                             'shift_binary_mask': 0.15,
                             'multiplex_shift_binary_mask2':0.2}

# [1] params choose
# result_path = '/results/exp2_mask_comparison/tmp'
scale = '512'
Cr = 10
gaussian_noise_levels = [5]
poisson_noise = False
gamma = 0.03
# mask_name = 'binary_mask'

# test_algo_flag = 'gaptv'
# test_algo_flag = 'gaptv+fastdvdnet'
# test_algo_flag = test_algo_flag_all[5]

mask_names = ['multiplex_shift_binary_mask2']
# orig_names = ['Jockey']
# orig_names = ['YachtRide']
orig_names = ['football']
# orig_names = ['hummingbird','ReadySteadyGo', 'Jockey', 'YachtRide']
# orig_names = orig_name_all
test_algo_flags = ['gapfastdvdnet','admmfastdvdnet', 'gaptv+fastdvdnet','admmtv+fastdvdnet']
# scales = ['512']
# Crs = [10]


show_res_flag = 1
save_res_flag = 1
log_result_flag = 1


iframe = 0                 # from which frame of meas to recon            
nframe = 1       # how many frame of meas to recon [img_num//Cr ]

sigma1 = 0     # scaler
iter_max1 = 65    
iter_max2 = [10, 85]
sigma2 = [50/255, 25/255]
# result_path = '/results/exp3_mask_comparison_noise/tmp/1'
print(' 1 -------------------------')
# [2] run
for test_algo_flag in test_algo_flags:
    for gaussian_noise_level in gaussian_noise_levels:
        for mask_name in mask_names:
            for orig_name in orig_names:
                result_path = '/results/exp3_mask_comparison_noise/tmp_admm/'+mask_name
                cli_run('pnp_sci_video_orig_simuexp3.py', orig_name, scale, Cr, mask_name, test_algo_flag,
                result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
                show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag, 
                gaussian_noise_level=gaussian_noise_level, poisson_noise=poisson_noise, gamma=gamma,
                tv_weight = opti_tv_weight_table_exp2[mask_name], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)
