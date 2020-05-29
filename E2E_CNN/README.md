# E2E_CNN_SCI
>  End to End CNN for Reconstruction of Snapshot Compressive Imaging

This repository contains the  codes modified from https://github.com/mq0829/DL-CACTI, which is designed for the paper **Deep Learning for Video Compressive Sensing** (***APL Photonics 5, 030801 (2020)***) by Mu Qiao*, Ziyi Meng*, Jiawei Ma, [Xin Yuan](https://www.bell-labs.com/usr/x.yuan) (*Equal contributions). [[pdf\]](https://aip.scitation.org/doi/pdf/10.1063/1.5140721?download=true) [[doi\]](https://aip.scitation.org/doi/10.1063/1.5140721)



# Structure of directories

- E2E_CNN_simu: The models, codes and results
- data_simu: The simulation data for training or testing. The input data is the scene ground truth ('orig') and mask ('mask'). And when training or testing, the coded measurement ('meas') will be generated automatically with 'orig' (rescale to 0-1 first) and 'mask'.
- data_meas: The simulation data for testing. The input data is the coded measurement ('meas') and mask ('mask')
- data_raw: Row dataset



# Data format

- 'orig': int, 0-255, , H\*W\*Compressive_ratio; '.mat' filetype; variable_name(key) = 'patch_save'
- 'mask': float 0-1|binary; '.mat' filetype; variable_name(key) = 'mask'
- 'meas': float; '.mat' filetype; variable_name(key) = 'meas'



# Usage 

## Environment

- Tensorflow-gpu==1.13.1 (conda install tensorflow-gpu=1.13.1)
- numpy, yaml, scipy, hdf5storage, matplotlib, math

## Training

1. put the ground truth (orig) datasets for training and validation in 'data_simu/training_truth' and 'data_simu/valid_truth', respectively.
2. modify configurations, like batch_size, learning rate, etc. in 'E2E_CNN_simu/Model/Config.yaml' . Particularly, If you have a pre-trained model for finetuning, specify it's path(like, 'Result/Model-Config/Decoder-T0519121103-D0.10L0.001-RMSE/models-0.2041-117480') in the 'model_filename' item of the 'Config.yaml' file)
3. run 'E2E_CNN_simu/train.py' to train the network
4. the result will be saved in 'E2E_CNN_simu/Result/Model-Config/'

## Testing

1. use ’orig‘ as the test input data：
   - Put the ’orig‘  in 'data_simu/testing_truth/', and put the 'mask' in 'data_simu/'
   - Open and modify 'E2E_CNN_simu/test_orig.py' according to the instructions in the beginning of the code and then run it.
   - The results will be saved as 'E2E_CNN_simu/Result/Validation-Result/Test_orig_result_i.mat'
2. use ’meas‘ as the test input data：
   - Put the data in 'data_meas/meas/', and put the 'mask' in 'data_mask/mask/'
   - Open and modify 'E2E_CNN_simu/test_meas.py' according to the instructions in the beginning of the code and then run it.
   - The results will be saved as 'E2E_CNN_simu/Result/Validation-Result/Test_meas_result_i.mat'



# Reference

**Deep Learning for Video Compressive Sensing** (***APL Photonics 5, 030801 (2020)***) by Mu Qiao*, Ziyi Meng*, Jiawei Ma, [Xin Yuan](https://www.bell-labs.com/usr/x.yuan) (*Equal contributions). [[pdf\]](https://aip.scitation.org/doi/pdf/10.1063/1.5140721?download=true) [[doi\]](https://aip.scitation.org/doi/10.1063/1.5140721)

 https://github.com/mq0829/DL-CACTI

https://github.com/Phoenix-V/tensor-admm-net-sci