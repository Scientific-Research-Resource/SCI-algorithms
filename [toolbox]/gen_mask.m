%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	generate E2E_CNN mask data
%	generate binary/gray/combined mask with the given size.
%
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-05-17
%   Last Modified:  Zhihong Zhang, 2020-05-17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% setting
% path setting
% root path
root_dir = 'E:\project\CACTI\simulation\CI algorithm\E2E_CNN\data_simu\mask\';
mask_name = 'combine_binary_mask_256_10f.mat';
mask_info_name = 'combine_binary_mask_256_10f_info.mat';


% param setting
% mask_size = [512 512]; 
mask_size = [256 256]; 
Cr = 10; % compressive ratio of snapshot
% save name for a Cr patch
mask_key = 'mask';
mask_type = 1; % 1-binary mask, 2-gray mask
combine_mask_flag = 1; % 0-not combine mask, 1-combine mask

%% generate mask
if mask_type==1
	% 1-binary mask
	init_mask = binary_mask([mask_size Cr]);
elseif  mask_type==2
	% 2-gray mask
	init_mask = gray_mask([mask_size Cr]);
end

if combine_mask_flag==0
	combine_matrix = eye(Cr); % combine0: non-combined
elseif combine_mask_flag==1
	combine_matrix = single(binary_mask(Cr));		% combine1
end

non_norm_mask = combine_mask(init_mask,combine_matrix);
mask = non_norm_mask./max(non_norm_mask, [],'a');

% histgram and statistic
figure,hist(init_mask(:),10), title('init mask distribution')
figure,hist(mask(:),10), title('combined mask distribution')
rank_combine_matrix = rank(combine_matrix)
mean_mask = mean(mask,'a')

%% save
save([root_dir mask_name], 'mask')
save([root_dir mask_info_name], 'init_mask', 'combine_matrix', 'non_norm_mask', 'mask')
disp(['mask saved to:' root_dir mask_name])