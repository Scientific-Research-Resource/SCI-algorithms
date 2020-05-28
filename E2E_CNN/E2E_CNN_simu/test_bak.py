from __future__ import absolute_import

import tensorflow as tf
import yaml
import os
import h5py

from Model.Decoder_Handler import Decoder_Handler

config_filename = './Model/Config.yaml'

def main():
    ave_folder,ave_config = 'Decoder-T0427184230-D0.10L0.010-RMSE/','config_45.yaml'
    # ave_folder,ave_config = 'Decoder-T0510154141-D0.10L0.010-RMSE/','config_311.yaml'
    
    folder_id,config_id = ave_folder,ave_config
    with open(config_filename) as handle:
        model_config = yaml.load(handle,Loader=yaml.FullLoader)  

    data_name = []
    data_name.append(os.path.join(os.path.abspath('..'), model_config['category']))
    data_name.append(os.path.join(os.path.abspath('..'), model_config['category_valid']))
    data_name.append(os.path.join(os.path.abspath('..'), model_config['category_test']))
    log_dir = os.path.join(os.path.abspath('.'),model_config['result_dir'],model_config['result_model'],folder_id)

    with open(os.path.join(log_dir, config_id)) as handle:
        model_config = yaml.load(handle,Loader=yaml.FullLoader)
    if model_config['mask_name'] == 'Original':
        mask_name = None
    else:
        mask_name = os.path.join(os.path.abspath('..'), model_config['category_mask'], model_config['mask_name'])
        
    dataset_name = (data_name,mask_name)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        Cube_Decoder = Decoder_Handler(dataset_name=dataset_name, model_config=model_config, sess = sess, is_training=False)
        Cube_Decoder.test()

if __name__ == '__main__':
    main()
    
    
    
    