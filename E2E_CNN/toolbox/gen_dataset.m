%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Generate Dataset for E2E_CNN
%	with data augmentation
%
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-05-17
%   Last Modified:  Zhihong Zhang, 2020-05-17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% setting
% path setting
root_dir = 'E:\project\CACTI\simulation\CI algorithm\E2E_CNN\';
% 480p resolution - 90 videos, 6208 images
data_raw_dir = 'data_raw\DAVIS-2017-Unsupervised-trainval\480p\';
% full resolution - 90 videos, 6208 images
% data_raw_dir = 'data_raw\DAVIS-2017-Unsupervised-trainval\full_resolution\'; 
data_raw_filetype = '*.jpg';

% save_path
% train_dir = 'data_simu\training_truth\';
% test_dir = 'data_simu\testing_truth\';
% valid_dir = 'data_simu\valid_truth\';
train_dir = 'test\1\';
test_dir = 'test\2\';
valid_dir = 'test\3\';

% param setting
Cr = 10; % compressive ratio of snapshot
% save_img_sz = [512 512]; % patch size
save_img_sz = [256 256]; % patch size
var_name = 'patch_save'; % save name for a Cr patch

% total num: 90 videos, 6208 images
trainset_num = 85; % number of videos used to generate training set
testset_num = 3;
validset_num = 2;

% opt
opt.data_startpoint_step = 5;
opt.image_sampling_step = 1;
opt.sampling_times_per_video = 5;
opt.data_aug_flip_flag = 1;
opt.data_aug_rot_flag = 1;



%% generating

% generate train/test/valid opt
train_opt = opt;
train_opt.dataset_start_point = 1;
train_opt.dataset_end_point = trainset_num;

test_opt = opt;
test_opt.dataset_start_point = trainset_num+1;
test_opt.dataset_end_point = trainset_num+testset_num;

valid_opt = opt;
valid_opt.dataset_start_point = trainset_num+testset_num+1;
valid_opt.dataset_end_point =  trainset_num+testset_num+validset_num;

% generating trainning set
disp('----- generate train set -----')
gen_multiframe_dataset([root_dir data_raw_dir], data_raw_filetype, Cr,...
	save_img_sz, var_name, [root_dir train_dir], train_opt)

% generating test set
disp('----- generate test set -----')
gen_multiframe_dataset([root_dir data_raw_dir], data_raw_filetype, Cr,...
	save_img_sz, var_name, [root_dir test_dir], test_opt)

% generating valid set
disp('----- generate valid set -----')
gen_multiframe_dataset([root_dir data_raw_dir], data_raw_filetype, Cr,...
	save_img_sz, var_name, [root_dir valid_dir], valid_opt)

disp('----- all dataset generated! -----')