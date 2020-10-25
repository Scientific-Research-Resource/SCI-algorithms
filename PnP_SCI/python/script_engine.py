import os
import numpy as np

#%%
# [0] config & params
## engine choose
# engine_flag = 'gaptv_finetune'
engine_flag = 'gaptv_cacti'
# engine_flag = 'pnp-cacti'

## params setting
orig_names = ['football', 'hummingbird', 'ReadySteadyGo', 'Jockey', 'YachtRide']   
scales = ['256', '512', '1024']
Crs = [10,20]

opti_tv_weight_table = {'256_Cr10': 0.20,
                   '256_Cr20': 0.15,
                   '512_Cr10': 0.30,
                   '512_Cr20': 0.1,
                   '1024_Cr10': 0.,
                   '1024_Cr20': 0.}
        
# [1] gaptv_finetune
if engine_flag=='gaptv_finetune':   
    tv_weights = np.arange(0.05, 0.5, 0.05)
    for scale in scales:
        for orig_name in orig_names:
            for Cr in Crs:
                for tv_weight in tv_weights:
                    print("python pnp_sci_video_orig_exp.py \
                        --orig_name {} --scale {} --Cr {} --tv_weight {:.2f}"
                        .format(orig_name, scale, Cr, tv_weight))
                    os.system("python pnp_sci_video_orig_exp.py \
                        --orig_name {} --scale {} --Cr {} --tv_weight {:.2f}"
                        .format(orig_name, scale, Cr, tv_weight))
                    
# [2] gaptv_cacti
if engine_flag == 'gaptv_cacti':
    for scale in scales:
        for orig_name in orig_names:
            for Cr in Crs:
                print("python pnp_sci_video_orig_exp.py \
                --orig_name {} --scale {} --Cr {} --tv_weight {:.2f}"
                .format(orig_name, scale, Cr, opti_tv_weight_table[scale+'_Cr'+str(Cr)]))
            os.system("python pnp_sci_video_orig_exp.py \
                --orig_name {} --scale {} --Cr {} --tv_weight {:.2f}"
                .format(orig_name, scale, Cr, opti_tv_weight_table[scale+'_Cr'+str(Cr)]))               


# [3] pnp-cacti
if engine_flag == 'pnp-cacti':
    pass