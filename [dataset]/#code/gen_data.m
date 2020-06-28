%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	generate CACTI data
%
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-06-26
%   Last Modified:  Zhihong Zhang, 2020-06-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% flags & params
% control flags
data_saving_flag = 1; % 1-save compact data, 2-save mask/meas/orig separately(ToDo)
img_resize_flag = 0; % resize 'orig' if it's not consistent with 'img_size'
show_eval_info = 1;
data_normalize_flag = 1; % normalize mask and meas [X./max(mask,'all)]

% params
Cr = 10; % compressive ratio
img_size = [512 512];
% img_size = [256 256];
mask_type = 0; % 0-load, 1-binary, 2-gray
combine_mask_type = 0; % 0-not combine mask, 1-random combine mask, 2-specific combine mask

% image path
save_dataset_folder = 'E:\project\CACTI\SCI algorithm\[dataset]\#test\data\combine_binary_mask_512_10f\bm_rescale_512_10f\';
save_dataset_name = 0; % default=0, save as 'data_[orig_name]'
% save_dataset_name = test.mat; 

load_mask_path = "E:\project\CACTI\SCI algorithm\[dataset]\#test\mask\combine_binary_mask_512_10f.mat";
% load_dataset_folder = 'E:\project\CACTI\SCI algorithm\[dataset]\#benchmark\orig\bm_256_10f\';
load_dataset_folder = 'E:\project\CACTI\SCI algorithm\[dataset]\#test\orig\bm_rescale_512_10f\';
load_dataset_names = 'all';
% load_dataset_name = {'aerial.mat'};
% load_dataset_name = {'crash.mat'};
% load_dataset_name = {'drop.mat'};
% load_dataset_name = {'kobe.mat'};
% load_dataset_name = {'runner.mat'};
% load_dataset_name = {'traffic.mat'};

% auto gen file names
if strcmp(char(load_dataset_names), 'all')
    files = dir([load_dataset_folder '*.mat']);
    load_dataset_names = {files.name};  
end

%% generate masks
if mask_type == 0
	% load mask
	% init_mask = load('init_test.mat', 'mask');
	% init_mask = load('512scale_traffic_cacti_simu.mat', 'mask');
	load_mask = load(load_mask_path, 'mask');
	load_mask = load_mask.mask;
	init_mask = load_mask(:,:,1:Cr);
	init_mask = single(init_mask);
elseif mask_type == 1
	% generate binary mask
	init_mask = single(binary_mask([img_size,Cr]));
elseif mask_type == 2
	% generate gray mask
	init_mask = gray_mask([img_size,Cr]);
end

% show eval info
if show_eval_info
	figure,hist(init_mask(:),10), title('mask distribution')
	min_mask = min(init_mask,[],'all');
	mean_mask = mean(init_mask,'all');
	max_mask = max(init_mask,[],'all');
	disp(['mask - min, mean, max: ' num2str(min_mask) ', ' num2str(mean_mask) ', ' num2str(max_mask)]);
end

% combine_matrix
if combine_mask_type==0
	% combine0: non-combined
	combine_matrix = eye(Cr); 
elseif combine_mask_type==1
	% combine1:random combine mask
	combine_matrix = single(binary_mask(Cr));
	% evaluate
	rank_combine_matrix = rank(combine_matrix);
	disp(['rank of combine_matrix: ' num2str(rank_combine_matrix)]);
	imshow(combine_matrix)
	disp('If ok, press any key to continue, or press Ctrl+C to stop...')
	pause		
elseif combine_mask_type==2
	% combine2: specific combine mask
end

if combine_mask_type==0
	mask = init_mask;
else
	% combine mask 
	non_norm_mask = combine_mask(init_mask,combine_matrix);
	mask = non_norm_mask./max(non_norm_mask, [],'a');		

	% show eval info
	if show_eval_info
		figure,hist(mask(:),10), title('combine mask distribution')
		min_mask = min(mask,[],'all');
		mean_mask = mean(mask,'a');
		max_mask = max(mask,[],'all');
		disp(['combine mask - min, mean, max: ' num2str(min_mask) ', ' num2str(mean_mask) ', ' num2str(max_mask)]);
	end

end

disp('----------')


%% processing
for load_file_name = load_dataset_names
    %% load orig file
    file_name = char(load_file_name);
    orig = load([load_dataset_folder file_name], 'orig');
    orig = orig.orig;
    orig = orig(:,:,1:Cr);
    % reszie
    if img_resize_flag && (any(size(orig(:,:,1)) ~= img_size, 'all'))
        orig =imresize(orig, img_size);
    end
    
    
	% orig info
	if show_eval_info
		min_orig = min(orig,[],'all');
		mean_orig = mean(orig,'a');
		max_orig = max(orig,[],'all');
		disp(['orig - min, mean, max: ' num2str(min_orig) ', ' num2str(mean_orig) ', ' num2str(max_orig)]);
	end
	
	



    %% coding frame
    coded_frame = single(mask).*single(orig);
    meas = sum(coded_frame, 3);
	
	% show eval info
	if show_eval_info
		min_meas = min(meas,[],'all');
		mean_meas= mean(meas,'a');
		max_meas = max(meas,[],'all');
		disp(['meas - min, mean, max: ' num2str(min_meas) ', ' num2str(mean_meas) ', ' num2str(max_meas)]);
	end    
    
	
    %% data normalize
    if data_normalize_flag
        mask_max = max(mask,[],'a');
        mask = mask./ mask_max;
        meas = meas./ mask_max;
    end

    %% data saving
	save_dataset_name = ['data_' file_name];
	
	if data_saving_flag==1
        % save dataset
		if combine_mask_type==0
			save([save_dataset_folder save_dataset_name],'orig','mask','meas')
		else
			save([save_dataset_folder save_dataset_name],'combine_matrix','init_mask', 'non_norm_mask', 'orig','mask','meas')
		end
	end
	
	disp(['finish: ' file_name])
end

disp(['dataset saved to: ' save_dataset_folder])