function gen_multiframe_dataset(raw_data_dir, raw_data_filetype, frame_num, img_sz, var_name, save_dir, opt)
%GEN_MULTIFRAME_DATASET generate '.mat' video segment datasets from the given videos(image sequence).
%	Each data item has a fixed image size and contains a fixed number of frames.
% 
%   Input:
%   --------
%   - raw_data_dir: path of row data directory, char. In this directory, there are
%     sub-directories. Each sub-directory contains a video (image sequence).
% 
%   - raw_data_filetype: filetype of images in the sub-directories, char,
%     e.g. '*.jpg'
% 
%	- frame_num: the amount of frames containded in the data item to
%	  generate, int scalar
% 
%	- img_sz: size of the image frames containded in the data item to
%	  generate, int 2D vector
% 
%	- var_name: variable name of the data item when saving it to a '.mat'
%	  file, char.
% 
%	- save_dir: saving directory of the generated data items, char.
% 
%	- opt: some detail options when generating data items, struct,
%	  including:
% 		opt.dataset_start_point: the first sub directory index to use, int,
% 		default = 1
% 
% 		opt.dataset_end_point: the last sub directory index to use, int,
% 		default = all files amount
% 
% 		opt.data_startpoint_step: the interval of the sampling start points
% 		in a video, int, default = frame_num (non-overlap between data
% 		items).
% 
% 		opt.image_sampling_step: sampling step length when extracting frame from a video.
% 		int, default=1
% 
% 		opt.sampling_times_per_video: sampling times per video, the samping area
% 		is randomly generated with fixed size('img_sz'). int, default=1
% 		
% 		opt.data_aug_flip_flag: whether to use flip(left-right and up-down)
% 		data augmentation, logical, default=0
% 		opt.data_aug_rot_flag: whether to use rotataion(2 random degrees)
% 		data augmentation, logical, default=1
% 
%   Note£º
%   --------
%	Sampling: 
%	1st_data_item_samping_idx = 1 : image_sampling_step : 1+image_sampling_step*(frame_num-1);
%	2nd_data_item_samping_idx = 1+data_startpoint_step : image_sampling_step : 
%								1+data_startpoint_step+image_sampling_step*(frame_num-1);
%	...
%	nth_data_item_samping_idx = n+data_startpoint_step : image_sampling_step : 
%								n+data_startpoint_step+image_sampling_step*(frame_num-1);
% 
%   Info£º
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-05-17
%   Last Modified:  Zhihong Zhang <z_zhi_hong@163.com>, 2020-05-17
%   
%   Copyright 2020 Zhihong Zhang

%% input checking
if ~isfield(opt, 'dataset_start_point')
	dataset_start_point = 1;
else
	dataset_start_point = opt.dataset_start_point;
end

if ~isfield(opt, 'dataset_end_point')
	dataset_end_point = 'end';
else
	dataset_end_point = opt.dataset_end_point;
end

if ~isfield(opt, 'data_startpoint_step')
	data_startpoint_step = frame_num;
else
	data_startpoint_step = opt.data_startpoint_step;
end

if ~isfield(opt, 'image_sampling_step')
	image_sampling_step = 1;
else
	image_sampling_step = opt.image_sampling_step;
end

if ~isfield(opt, 'sampling_times_per_video')
	sampling_times_per_video = 1;
else
	sampling_times_per_video = opt.sampling_times_per_video;
end

if ~isfield(opt, 'data_aug_rot_flag')
	data_aug_rot_flag = 0;
else
	data_aug_rot_flag = opt.data_aug_rot_flag;
end

if ~isfield(opt, 'data_aug_flip_flag')
	data_aug_flip_flag = 0;
else
	data_aug_flip_flag = opt.data_aug_flip_flag;
end

%% get sub dirs which contains video frames
sub_dirs = dir(raw_data_dir);
sub_dir_names = {sub_dirs(3:end).name};
sub_dir_num = length(sub_dir_names);

% default value for 'dataset_end_point'
if ischar(dataset_end_point)
	dataset_end_point = sub_dir_num;
end

tic;
using_raw_dataset_num = dataset_end_point - dataset_start_point+1; % number of raw videos(sub dirs) to use
sub_dir_i = 0; % sub dir(video dir) index
for m = dataset_start_point:dataset_end_point
	% 'm': video index
	sub_dir_i = sub_dir_i+1;
    % image names in a video dir
    files =dir(fullfile(raw_data_dir,sub_dir_names{m},raw_data_filetype));
    files_name = {files.name};
    files_num = length(files_name);    
	
	% sampling, cropping and saving
	data_i = 0; % data item index
    for n= 1:data_startpoint_step:files_num
		% 'n': image segment start point index
		data_i = data_i+1; 
		start_i = n;
		end_i = n+(frame_num-1)*image_sampling_step;
		if end_i > files_num
			% out of max number
			break;
		end
		
		% generating ROI rectangle's left-up vertex (random)
		tmp_img1 = imread(fullfile(raw_data_dir,sub_dir_names{m},files_name{1}));
		tmp_size = size(tmp_img1);
		row_min = 1; row_max = tmp_size(1)-img_sz(1);
		col_min = 1; col_max = tmp_size(2)-img_sz(2);

		ROI_p = zeros(sampling_times_per_video,2);
		ROI_p(:,1) = randi([row_min, row_max], [sampling_times_per_video, 1]);
		ROI_p(:,2) = randi([col_min, col_max], [sampling_times_per_video, 1]);		
	
		
		img_segs = zeros([img_sz, frame_num, sampling_times_per_video]);  % image segments
		if data_aug_flip_flag
			% image segments for 'flip' data augmentation
			img_segs_flip_ud = zeros([img_sz, frame_num, sampling_times_per_video]);  
			img_segs_flip_lr = zeros([img_sz, frame_num, sampling_times_per_video]);
		end
		
		if data_aug_rot_flag
		% image segments for 'rotate' data augmentation
			img_segs_rot1 = zeros([img_sz, frame_num, sampling_times_per_video]); 
			img_segs_rot2 = zeros([img_sz, frame_num, sampling_times_per_video]);  
			% generating random rotate degrees
			rot_degree1 = randi([5,180]);
			rot_degree2 = randi([180,360]);
		end
		
		frame_i = 0; % segment frame index
		for i = start_i:image_sampling_step:end_i
			% 'i': image index in a image segment
			% read image and convert to gray
			tmp_img = imread(fullfile(raw_data_dir,sub_dir_names{m},files_name{i}));
			tmp_img_gray = rgb2gray(tmp_img);
			
			frame_i = frame_i+1;
			% crop and data augmentation
			for j = 1:sampling_times_per_video
				% 'j': sampling index in a image
				% crop
				tmp_crop = imcrop(tmp_img_gray,[ROI_p(j,2) ROI_p(j,1) img_sz(2)-1 img_sz(1)-1]);
				img_segs(:,:,frame_i,j) = double(tmp_crop);
				
				% 'flip' data augmentation
				if data_aug_flip_flag
					tmp_crop_flip_ud = flipud(tmp_crop);
					tmp_crop_flip_lr = fliplr(tmp_crop);
					img_segs_flip_ud(:,:,frame_i,j) = tmp_crop_flip_ud;
					img_segs_flip_lr(:,:,frame_i,j) = tmp_crop_flip_lr;
				end
				
				% 'rotate' data augmentation
				if data_aug_rot_flag
					tmp_crop_rot1 = imrotate(tmp_crop,rot_degree1,'crop');
					tmp_crop_rot2 = imrotate(tmp_crop,rot_degree2,'crop');
					img_segs_rot1(:,:,frame_i,j) = tmp_crop_rot1;
					img_segs_rot2(:,:,frame_i,j) = tmp_crop_rot2;
				end
			end
		end
		
		% save
		for j = 1:sampling_times_per_video
			% save data
			eval([var_name '= img_segs(:,:,:,j);']);
			save([save_dir sub_dir_names{m} num2str(data_i) '_' num2str(j) '.mat'], var_name)
			
			% save augmented data
			if data_aug_flip_flag
				eval([var_name '= img_segs_flip_lr(:,:,:,j);']);
				save([save_dir sub_dir_names{m} num2str(data_i) '_' num2str(j)  '_fliplr.mat'], var_name)
				eval([var_name '= img_segs_flip_ud(:,:,:,j);']);
				save([save_dir sub_dir_names{m} num2str(data_i) '_' num2str(j) '_flipud.mat'], var_name)				
			end			
			if data_aug_rot_flag
				eval([var_name '= img_segs_rot1(:,:,:,j);']);
				save([save_dir sub_dir_names{m} num2str(data_i) '_' num2str(j) '_rot1.mat'], var_name)
				eval([var_name '= img_segs_rot2(:,:,:,j);']);
				save([save_dir sub_dir_names{m} num2str(data_i) '_' num2str(j) '_rot2.mat'], var_name)					
			end
			
		end
		
	end
	
    if mod(m,ceil(using_raw_dataset_num/40))==0
		t = toc;
        disp(['time: ' num2str(t) 's;  ' num2str(100*sub_dir_i/using_raw_dataset_num) '% done!'])
    end
end

disp(['datasets saved to: ' save_dir])
end
