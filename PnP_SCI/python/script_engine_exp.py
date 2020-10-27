import os
import numpy as np
from my_tools import cli_run
#%%
# [0] config & params
## engine choose
# engine_flag = 'gaptv_finetune'
# engine_flag = 'gaptv_cacti'
engine_flag = 'pnp-cacti'



## params setting
mask_name = 'multiplex_shift_binary_mask'
orig_names = ['football', 'hummingbird', 'ReadySteadyGo', 'Jockey', 'YachtRide'] 
scales = ['256', '512', '1024']
Crs = [10,20]
# test_algo_flags = ['all', 'gaptv', 'admmtv', 'gapffdnet', 'admmffdnet', 
#               'gapfastdvdnet', 'admmfastdvdnet', 'gaptv+ffdnet', 'gaptv+fastdvdnet']
test_algo_flag = 'gaptv' # fastdvdnet

iter_max1 = 20     # scaler
sigma1 = -1     # scaler
iter_max2 = [60, 100, 150]  # list
sigma2 = [100/255, 50/255, 25/255] #list

result_path = '/results/tmp' #
show_res_flag = 0
save_res_flag = 0

iframe = 0                 # from which frame of meas to recon            
nframe = 2       # how many frame of meas to recon [img_num//Cr ]

MAXB = 255
img_num=48
opti_tv_weight_table = {'256_Cr10': 0.20,
                   '256_Cr20': 0.15,
                   '512_Cr10': 0.25,
                   '512_Cr20': 0.10,
                   '1024_Cr10': 0.10,
                   '1024_Cr20': 0.10}
      
# [1] gaptv_finetune
if engine_flag=='gaptv_finetune':   
    tv_weights = np.arange(0.05, 0.5, 0.05)
    for scale in scales:
        for orig_name in orig_names:
            for Cr in Crs:
                for tv_weight in tv_weights:
                    cli_run(orig_name, scale, Cr, mask_name, test_algo_flag,
                    result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB, 
                    show_res_flag = show_res_flag, save_res_flag =  save_res_flag , 
                    tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)
                    
# [2] gaptv_cacti
if engine_flag == 'gaptv_cacti':
    for scale in scales:
        for Cr in Crs:
            for orig_name in orig_names:
                    cli_run(orig_name, scale, Cr, mask_name, test_algo_flag,
                    result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB, 
                    show_res_flag = show_res_flag, save_res_flag =  save_res_flag , 
                    tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)


# [3] pnp-cacti
if engine_flag == 'pnp-cacti':
    for scale in scales:
        for Cr in Crs:
            for orig_name in orig_names:
                    cli_run(orig_name, scale, Cr, mask_name, test_algo_flag,
                    result_path = result_path, iframe = iframe, nframe = img_num//Cr, MAXB = MAXB, 
                    show_res_flag = show_res_flag, save_res_flag =  save_res_flag , 
                    tv_weight = opti_tv_weight_table[scale+'_Cr'+str(Cr)], iter_max1 = iter_max1, sigma1 = sigma1, iter_max2 = iter_max2, sigma2 = sigma2)
