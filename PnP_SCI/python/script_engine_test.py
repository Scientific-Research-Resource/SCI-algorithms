import os
import numpy as np
from my_tools import cli_run
# %%
# [0] pre-params setting
orig_name_all = ['football', 'hummingbird',
                 'ReadySteadyGo', 'Jockey', 'YachtRide']
scale_all = ['256', '512', '1024']
Cr_all = [10, 20]

test_algo_flag_all = ['all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet',
                      'gapfastdvdnet', 'admmfastdvdnet', 'gaptv+ffdnet', 'gaptv+fastdvdnet']

MAXB = 255
img_num=48
opti_tv_weight_table = {'256_Cr10': 0.20,
                        '256_Cr20': 0.15,
                        '512_Cr10': 0.25,
                        '512_Cr20': 0.10,
                        '1024_Cr10': 0.10,
                        '1024_Cr20': 0.10}


# [1] params choose
result_path = '/results/exp2_mask_comparison/tmp'

mask_name = 'binary_mask'
# mask_name = 'multiplex_shift_binary_mask'
# orig_names = ['Jockey']
# orig_names = ['YachtRide']
orig_names = ['football']
# orig_names = ['hummingbird','ReadySteadyGo', 'Jockey', 'YachtRide']
# orig_names = orig_name_all
scales = ['512']
Crs = [10]


show_res_flag = 1
save_res_flag = 1
log_result_flag = 1
# test_algo_flag = 'gaptv'
test_algo_flag = 'gaptv+fastdvdnet'
# test_algo_flag = test_algo_flag_all[5]

iframe = 0                 # from which frame of meas to recon            
nframe = 1       # how many frame of meas to recon [img_num//Cr ]

sigma1 = 0     # scaler
iter_max1 = 5   
sigma2 = [100/255, 50/255, 25/255]
iter_max2 = [10, 30, 205]   
print(' 1 -------------------------')
# [2] run
for scale in scales:
    for Cr in Crs:
        for orig_name in orig_names:
            cli_run('pnp_sci_video_orig_simuexp2.py', orig_name, scale, Cr, mask_name, test_algo_flag,
            result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
            show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag,
            tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)

# iter_max1 = 5   
# sigma2 = [100/255, 50/255, 25/255]
# iter_max2 = [10, 15, 220]         
# print(' 2 -------------------------')
# # [2] run
# for scale in scales:
#     for Cr in Crs:
#         for orig_name in orig_names:
#             # result_path = '/results/exp1_multiscale/PnP-tv-fastdvdnet/'+scale
#             cli_run('pnp_sci_video_orig_simuexp2.py',orig_name, scale, Cr, mask_name, test_algo_flag,
#             result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
#             show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag,
#             tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)

         
# iter_max1 = 5   
# sigma2 = [100/255, 50/255, 25/255]
# iter_max2 = [10, 30, 205]  
# print(' 3 -------------------------')
# # [2] run
# for scale in scales:
#     for Cr in Crs:
#         for orig_name in orig_names:
#             # result_path = '/results/exp1_multiscale/PnP-tv-fastdvdnet/'+scale
#             cli_run('pnp_sci_video_orig_simuexp2.py',orig_name, scale, Cr, mask_name, test_algo_flag,
#             result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
#             show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag,
#             tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)