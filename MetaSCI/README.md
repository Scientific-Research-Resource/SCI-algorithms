# MetaSCI-CVPR2021

This code is for CVPR 2021 paper "MetaSCI: Scalable and Adaptive Reconstruction for Video Compressive Sensing" （modified by [Zhihong Zhang](https://github.com/dawnlh)).  Original code repository is located at https://github.com/xyvirtualgroup/MetaSCI-CVPR2021





### Directory

- `dataset\mask` : encoding mask, shape = [M N Cr NUM], i.e., NUM  [M N Cr] masks stacked along the last dimension.
- `dataset\orig` : original frames, shape = [M N Cr], i.e., Cr  [M N] frames stacked along the last dimension.



### Parameter

- `datadir`:  path for training set directory
- `maskpath`: path for encoding mask file



### Run

#### Training

1. Prepare the masks: divide the large-scale mask into small patches (w./w.o. overlap) with the same size. Save one of the patch for the base model's training, and save the other patches for the adapting models' training.
2. Train the base model with `main_MetaBaseModel_train.py` and one of the mask patch above (`main_MetaBaseModel_train_parallel.py` can be used for parallel training).
3. Finetune the base model above to get the adapting models with `main_MetaAdaptModel_train.py` and the other mask patches.

#### Test

1. Use `main_MetaBaseModel_test.py` and `main_MetaAdaptModel_test.py ` and corresponding checkpoints and masks to reconstruct the video.
