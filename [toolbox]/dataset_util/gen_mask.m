%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	generate CACTI mask
%	generate binary/gray/combined mask with the given size.
%
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-05-17
%   Last Modified:  Zhihong Zhang, 2020-10-21
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% setting
% envir init
addpath(genpath('E:\project\CACTI\SCI algorithm\[toolbox]'));

% path setting
% root path
root_dir = 'E:\project\CACTI\SCI algorithm\[dataset]\#benchmark\mask\tmp\';
mask_name = 'combine_shift_binary_mask_256_10f.mat';
mask_info_name = 'combine_shift_binary_mask_256_10f_info.mat';


% param setting
% mask_size = [512 512]; 
mask_size = [256 256]; 
Cr = 10; % compressive ratio of snapshot
blk_sz = [3,4]; % for shifting mask: shifting pixel range
% save name for a Cr patch
mask_key = 'mask';
mask_type = 4; % 1-binary init mask, 2-gray init mask, 3-load mask, 4-shift binary mask
init_mask_path = 'E:\project\CACTI\SCI algorithm\[dataset]\#benchmark\mask\binary_mask_256_10f.mat'; % for load init mask

combine_mask_flag = 2; % 0-not combine mask, 1-random combine mask, 2-specific combine mask

%% generate mask
% init mask
if mask_type==1
	% 1-binary mask
	init_mask = binary_mask([mask_size Cr]);
elseif  mask_type==2
	% 2-gray mask
	init_mask = gray_mask([mask_size Cr]);
elseif  mask_type==3
	% 3-load mask
	init_mask = load(init_mask_path);
	init_mask = init_mask.mask;
elseif  mask_type==4
	% 4-shfit mask
	src_mask = binary_mask(mask_size+5);
	init_mask = shift_mask(src_mask, mask_size, blk_sz, 'range');
	init_mask = init_mask(:,:,1:Cr);
end

% combine_matrix
if combine_mask_flag==0
	% combine0: non-combined
	combine_matrix = eye(Cr); 
elseif combine_mask_flag==1
	% combine1:random combine mask
	combine_matrix = single(binary_mask(Cr));	
elseif combine_mask_flag==2
	%combine2: specific combine mask
	combine_matrix = [1,1,1,0,1,0,0,1,0,0;
					  0,1,1,1,0,1,0,1,0,0;
					  0,0,1,1,1,1,1,0,0,0;
					  0,0,0,1,1,1,1,1,0,0;
					  0,0,0,0,1,1,1,1,1,0;
					  0,0,0,0,0,1,1,1,1,1;
					  1,0,0,0,0,0,1,1,1,1;
					  1,0,0,0,1,0,0,1,1,1;
					  1,1,1,0,0,0,0,0,1,1;
					  1,1,0,1,0,1,0,0,0,1];
end
% evaluate
rank_combine_matrix = rank(combine_matrix);
disp(['rank of combine_matrix: ' num2str(rank_combine_matrix)]);
imshow(combine_matrix)
disp('If ok, press any key to continue, or press Ctrl+C to stop')
pause

% combine mask 
non_norm_mask = combine_mask(init_mask,combine_matrix);
mask = non_norm_mask./max(non_norm_mask, [],'a');

% histgram and statistic
figure,hist(init_mask(:),10), title('init mask distribution')
figure,hist(mask(:),10), title('combined mask distribution')


%% save
save([root_dir mask_name], 'mask')
save([root_dir mask_info_name], 'init_mask', 'combine_matrix', 'non_norm_mask', 'mask')
disp(['mask saved to:' root_dir mask_name])