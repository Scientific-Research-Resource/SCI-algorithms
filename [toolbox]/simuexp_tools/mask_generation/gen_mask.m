%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	generate CACTI mask
%	generate binary/gray/multiplexd mask with the given size.
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
root_dir = '.\tmp\';

% param setting
% mask_size = [1024 1024]; 
% mask_size = [512 512]; 
mask_size = [256 256]; 
Cr = 10; % compressive ratio of snapshot
blk_sz = [3,4]; % for shifting mask: shifting pixel range; [3,4] for Cr=10; [4,5] for Cr=20
complex_num = 1;  % complexing num for rach mask
% save name for a Cr patch
mask_key = 'mask';
mask_type = 4; % 1-binary init mask, 2-gray init mask, 3-load mask, 4-shift binary mask
% init_mask_path = 'E:\project\CACTI\SCI algorithm\[dataset]\#benchmark\mask\binary_mask_256_10f.mat'; % for load init mask
init_mask_path = 'E:\project\CACTI\experiment\simulation\dataset\simu_data\gray\mask\tmp\init_binary_mask_256_12f.mat';
multiplex_mask_flag = 1; % 0-not multiplex mask, 1-random multiplex mask, 2-specific multiplex mask

mask_name = ['shift_binary_mask2_' num2str(mask_size(1)) '_' num2str(Cr) 'f.mat'];   % multiplex_ , shift, binary
mask_info_name = ['shift_binary_mask2_' num2str(mask_size(1)) '_' num2str(Cr) 'f_info.mat'];


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
	init_mask = init_mask.init_mask;
elseif  mask_type==4
	% 4-shfit mask
	src_mask = binary_mask(mask_size+10);  % generate source mask
	% src_mask = load(init_mask_path);      % load source mask
	% src_mask = src_mask.mask;	
	
	% 1 pixel shift for adjacent sub-aperture
	init_mask = shift_mask(src_mask, mask_size, blk_sz, 'range'); 
end

% multiplex_matrix
if multiplex_mask_flag==0
	% multiplex0: non-multiplexd
	multiplexing_matrix = eye(Cr); 
elseif multiplex_mask_flag==1
	% multiplex1:random multiplex mask
% 	multiplex_matrix = single(binary_mask(Cr));	
% 	multiplexing_matrix = multiplex_matrix([Cr, prod(blk_sz)],complex_num );
	multiplexing_matrix = 1/prod(blk_sz)*multiplex_matrix([Cr, prod(blk_sz)],complex_num );	
elseif multiplex_mask_flag==2
	%multiplex2: specific multiplex mask
	multiplexing_matrix = [1,1,1,0,1,0,0,1,0,0;
					  0,1,1,1,0,1,0,1,0,0;
					  0,0,1,1,1,1,1,0,0,0;
					  0,0,0,1,1,1,1,1,0,0;
					  0,0,0,0,1,1,1,1,1,0;
					  0,0,0,0,0,1,1,1,1,1;
					  1,0,0,0,0,0,1,1,1,1;
					  1,0,0,0,1,0,0,1,1,1;
					  1,1,1,0,0,0,0,0,1,1;
					  1,1,0,1,0,1,0,0,0,1]; % 50% throughtput
end
% evaluate
rank_multiplex_matrix = rank(multiplexing_matrix);
disp(['rank of multiplex_matrix: ' num2str(rank_multiplex_matrix)]);
imshow(multiplexing_matrix,[])
disp('If ok, press any key to continue, or press Ctrl+C to stop')
pause
disp('continue...')

% multiplex mask 
non_norm_mask = multiplex_mask(init_mask,multiplexing_matrix);
mask = non_norm_mask./max(non_norm_mask, [],'a');

% histgram and statistic
figure,hist(init_mask(:),10), title('init mask distribution')
figure,hist(mask(:),10), title('multiplexd mask distribution')


%% save
save([root_dir mask_name], 'mask')
save([root_dir mask_info_name], 'init_mask', 'multiplexing_matrix', 'non_norm_mask', 'mask')
disp(['mask saved to:' root_dir mask_name])