import os
import numpy as np

tv_weights = np.arange(0.05, 1.1, 0.05)
orig_names = ['football', 'messi', 'hummingbird', 'swinger', 
              'ReadySteadyGo', 'Jockey', 'YachtRide']
scales = ['256', '512', '1024']
Crs = [20, 10]

for orig_name in orig_names:
    for scale in scales:
        for Cr in Crs:
            for tv_weight in tv_weights:
                print("python pnp_sci_video_orig_exp.py \
                    --orig_name {} --scale {} --Cr {} --tv_weight {:.2f}"
                    .format(orig_name, scale, Cr, tv_weight))
                os.system("python pnp_sci_video_orig_exp.py \
                    --orig_name {} --scale {} --Cr {} --tv_weight {:.2f}"
                    .format(orig_name, scale, Cr, tv_weight))