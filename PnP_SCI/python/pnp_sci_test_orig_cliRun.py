import os
import numpy as np
from utils import cli_run
#%%
# [0] config & params
## engine choose
engine_flag = 'cli_test'
# engine_flag = 'gaptv_finetune'
# engine_flag = 'gaptv_cacti'
# engine_flag = 'pnp-cacti'

## params setting
mask_names = ['binary_mask', 'shift_binary_mask', 'multiplex_shift_binary_mask2']
# mask_name = 'multiplex_shift_binary_mask'
orig_names = ['football', 'hummingbird', 'ReadySteadyGo', 'Jockey', 'YachtRide'] 
# scales = ['256', '512', '1024']
scales = ['512']
Crs = [10]
# test_algo_flags = ['all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet', 
#               'gapfastdvdnet', 'admmfastdvdnet', 'gaptv+ffdnet', 'gaptv+fastdvdnet']
# test_algo_flag = 'gaptv' # fastdvdnet

iter_max1 = 100     # scaler
sigma1 = 0     # scaler
# iter_max2 = [60, 100, 150]  # list
# sigma2 = [100/255, 50/255, 25/255] #list

# result_path = '/results/tmp' #
show_res_flag = 0
save_res_flag = 0
log_result_flag = 0

iframe = 0                 # from which frame of meas to recon            
nframe = 1       # how many frame of meas to recon [img_num//Cr ]

MAXB = 255
img_num=48

opti_tv_weight_table_exp2 = {'256_Cr10': 0.20,   
                   '256_Cr20': 0.15,
                   '512_Cr10': 0.25,
                   '512_Cr20': 0.10,
                   '1024_Cr10': 0.10,
                   '1024_Cr20': 0.10}
#  opti_tv_weight_table[scale+'_Cr'+str(Cr)]


# [iter_max1, iter_max2, sigma2]
opti_sigma_iter_table_exp2 = {'256_Cr10': [20, [20, 30, 90], [100/255, 50/255, 25/255]],
                   '256_Cr20': [10, [20, 20, 200], [100/255, 50/255, 25/255]],
                   '512_Cr10': [65, [10, 85], [50/255, 25/255]],
                   '512_Cr20': [5, [10, 30, 205], [100/255, 50/255, 25/255]],
                   '1024_Cr10': [55, [10, 95], [50/255, 25/255]],
                   '1024_Cr20': [20, [25/255], [230]]}

# [0] cli_test
mask_name = 'binary_mask'
if engine_flag == 'cli_test':
    for scale in scales:
        for Cr in Crs:
            for orig_name in orig_names:
                root_dir = 'E:/project/CACTI/experiment/simulation'
                result_path = '/results/tmp'
                # result_path = '/results/exp1_multiscale/PnP-tv-fastdvdnet/'+scale
                cli_run('pnp_sci_test_orig_cli.py', orig_name, scale, Cr, mask_name, 'gaptv+fastdvdnet',
                root_dir=root_dir, result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB,
                show_res_flag = show_res_flag, save_res_flag =  save_res_flag , log_result_flag=log_result_flag,
                tv_weight = opti_tv_weight_table_exp2[scale+'_Cr'+str(Cr)], 
                iter_max1 = opti_sigma_iter_table_exp2[scale+'_Cr'+str(Cr)][0], sigma1 = sigma1, 
                iter_max2 = opti_sigma_iter_table_exp2[scale+'_Cr'+str(Cr)][1], sigma2 = opti_sigma_iter_table_exp2[scale+'_Cr'+str(Cr)][2])
                     
# [1] gaptv_finetune
# exp0:
# if engine_flag=='gaptv_finetune':   
#     tv_weights = np.arange(0.05, 0.5, 0.05)
#     for scale in scales:
#         for orig_name in orig_names:
#             for Cr in Crs:
#                 for tv_weight in tv_weights:
#                     result_path = '/results/exp1_multiscale/gaptv/Cr'+scale
#                     cli_run('pnp_sci_video_orig_simuexp2.py',orig_name, scale, Cr, mask_name, 'gaptv',
#                     result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB, 
#                     show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag, 
#                     tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1)

# exp2:
if engine_flag=='gaptv_finetune':   
    tv_weights = np.arange(0.05, 0.5, 0.05)
    
    for mask_name in mask_names:
        for tv_weight in tv_weights:
            for orig_name in orig_names:
                result_path = '/results/exp0_gaptv_finetune/' + mask_name
                cli_run('pnp_sci_video_orig_simuexp2.py',orig_name, scales[0], Crs[0], mask_name, 'gaptv',
                result_path = result_path, iframe = iframe, nframe = nframe, MAXB = MAXB, 
                show_res_flag = show_res_flag, save_res_flag =  save_res_flag, log_result_flag=log_result_flag, 
                tv_weight = tv_weight, iter_max1 = iter_max1)

              
# [2] gaptv_cacti
# if engine_flag == 'gaptv_cacti':
#     for scale in scales:
#         for Cr in Crs:
#             for orig_name in orig_names:
#                 result_path = '/results/exp1_multiscale/PnP-tv-fastdvdnet/'+scale
#                 cli_run('pnp_sci_video_orig_simuexp2.py',orig_name, scale, Cr, mask_name, 'gaptv',
#                 result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB, 
#                 show_res_flag = show_res_flag, save_res_flag =  save_res_flag , log_result_flag=log_result_flag,
#                 tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1)

if engine_flag == 'gaptv_cacti':
    for Cr in Crs:
        for scale in scales:
            for orig_name in orig_names:
                if Cr==10:
                    iter_max1 = 160
                    result_path = '/results/exp1_multiscale/gaptv/Cr10_iter160'
                elif Cr==20:
                    iter_max1 = 250
                    result_path = '/results/exp1_multiscale/gaptv/Cr20_iter250'                    
                cli_run('pnp_sci_video_orig_simuexp2.py',orig_name, scale, Cr, mask_name, 'gaptv',
                result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB, 
                show_res_flag = show_res_flag, save_res_flag =  save_res_flag , log_result_flag=log_result_flag,
                tv_weight = opti_tv_weight_table_exp2[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1)
                
# [3] pnp-cacti
if engine_flag == 'pnp-cacti':
    for scale in scales:
        for Cr in Crs:
            for orig_name in orig_names:
                result_path = '/results/exp1_multiscale/PnP-tv-fastdvdnet/'+scale
                cli_run('pnp_sci_video_orig_simuexp2.py', orig_name, scale, Cr, mask_name, 'gaptv+fastdvdnet',
                result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB, 
                show_res_flag = show_res_flag, save_res_flag =  save_res_flag , log_result_flag=log_result_flag,
                tv_weight = opti_tv_weight_table_exp2[scale+'_Cr'+str(Cr)], 
                iter_max1 = opti_sigma_iter_table_exp2[scale+'_Cr'+str(Cr)][0], sigma1 = sigma1, 
                iter_max2 = opti_sigma_iter_table_exp2[scale+'_Cr'+str(Cr)][1], sigma2 = opti_sigma_iter_table_exp2[scale+'_Cr'+str(Cr)][2])
