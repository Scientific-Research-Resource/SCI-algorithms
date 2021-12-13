%% data preprocess for real CACTI data (calibrate all masks in one time)
% Support: 
%	positive-only / positive-negetive mode
%	patch processing (border extended with black for non exact division)

clc, clear;

%% [0] params
root = 'E:\project\SCI_captioning\experiment\realexp\test\SCICAP_outdoor_Cr8_20211213';
img_sz = [2048,2560];
cs_rate = 16;
save_patch = 0; % save patches
only_positive_mask = 1; % save patches

% patch param (when only_positive_mask = 0)
x_min = 1;
y_min = 1;
x_max = img_sz(2);
y_max = img_sz(1);
overlap = 50;
ps = 256;

black_start = framePgroup-3; % idx in group 0
white_start = framePgroup-1; % idx in group 0

mask_dir_files = listdir(fullfile(root,mask_dir));

%% [1.0] params for mask proc
mask_dir = 'masks';
group_num = 20;
framePgroup = 36;  % 20 - Cr8pn; 36-Cr16pn
% framePgroup = 20;  % 20 - Cr8pn; 36-Cr16pn

%% [1.1] process mask background and illumination
% white/black mask
count = 0;
for n = 1:group_num
    white_id = white_start + n*framePgroup;
    white_mask = imread(fullfile(root,mask_dir,mask_dir_files(white_id)));
    black_id = black_start + n*framePgroup;
    black_mask = imread(fullfile(root,mask_dir,mask_dir_files(black_id)));
    
    if n == 1
        white_mask_sum = zeros(size(white_mask));
        black_mask_sum = zeros(size(black_mask));
    end
    white_mask_sum = white_mask_sum + double(white_mask);
    black_mask_sum = black_mask_sum + double(black_mask);
    count = count + 1;
end
white_mask_mean = white_mask_sum / count;
black_mask_mean = black_mask_sum / count;
white_debkg = double(white_mask_mean) - double(black_mask_mean);


%% [1.2] process calibrated masks
% positive masks
for m = 1:cs_rate
    count = 0;
    mask_start = (m-1)*2+1; % idx in group 0
    for n = 1:group_num
        count = count + 1;
        mask_id = mask_start + n*framePgroup;
        mask_t = imread(fullfile(root,mask_dir,mask_dir_files(mask_id)));
        if n == 1
           mask_sum = zeros(size(mask_t));
        end
        mask_sum = mask_sum + double(mask_t);
    end
    mask_mean = mask_sum / count;
    mask_ori(:,:,m) = mask_mean;
end

% negative masks
if only_positive_mask==0
	for m = 1:cs_rate
		count = 0;
		mask_start = (m-1)*2+2; % start from 0
		for n = 1:group_num
			count = count + 1;
			mask_id = mask_start + n*framePgroup;
			mask_t = imread(mask_dir_files(black_id(mask_id)));
			if n == 1
			   mask_sum = zeros(size(mask_t));
			end
			mask_sum = mask_sum + double(mask_t);
		end
		mask_mean = mask_sum / count;
		mask_ori(:,:,m+cs_rate) = mask_mean;
	end
end

if only_positive_mask
	mask_num = cs_rate;
else
	mask_num = cs_rate*2;
end

for n = 1:1:mask_num
    mask_debkg(:,:,n) = double(mask_ori(:,:,n)) - double(black_mask_mean);
    mask_deillum_t = double(mask_debkg(:,:,n) ./ white_debkg);
    mask_deillum(:,:,n) = mask_deillum_t;
    figure(19)
    imshow(mask_deillum_t)
    title(sprintf('mask%d',n))
end

% save mask
if save_patch==0
% save full mask
	% mask = mask_deillum;
	mask = mask_deillum(:,:,1:mask_num);
	mask_full_save_dir = fullfile(root,'process/full/mask/'); 
	mkdir(mask_full_save_dir)
	save([mask_full_save_dir 'mask_full.mat'],'mask') 
else
% save patch mask
	x_max = x_min + ceil((x_max - x_min + 1 - overlap) / (ps - overlap)) * (ps - overlap) + overlap - 1;
	y_max = y_min + ceil((y_max - y_min + 1 - overlap) / (ps - overlap)) * (ps - overlap) + overlap - 1;


	% padding to mask
	mask_deillum_full = ones(y_max,x_max,mask_num);
	mask_deillum_full(1:img_sz(1),1:img_sz(2),:) = mask_deillum;
	
	% crop whole mask into patches
	save_dir = fullfile(root,sprintf('process/patch/overlap_%d',overlap));
	save_dir_mask = fullfile(save_dir,'mask','all');
	mkdir(save_dir_mask)
	count = 0;
	figure(20)
	for y = 1:(ps-overlap):(y_max-ps+1)
		for x = 1:(ps-overlap):(x_max-ps+1)
			count = count + 1;
			mask = mask_deillum_full(y:y+ps-1, x:x+ps-1, :);
			disp(['saving patch - ', num2str(count)])
			mask_save_path = fullfile(save_dir_mask,sprintf('mask%s.mat',num2str(count,'%03d')));
			if only_positive_mask==1
				
				save(mask_save_path,'mask')
			else
				mask_p = mask(:,:, 1:cs_rate);
				mask_n = mask(:,:, cs_rate+1:cs_rate*2);
				save(mask_save_path,'mask_p','mask_n')
			end
			
			imshow(mask(:,:,1)), title(sprintf('mask_1st patch-%d',count));
			pause(0.1)
		end
	end
	disp('Mask Done')
end

%% [2.0] params for meas proc
meas_dir = 'person_cr8';
% idxs = ["x.tif"]; % [] for all
meas_names = []; % [] for all

%% [2.1] process meas background
black_meas_mean = double(ones(img_sz)) * 2000;

%% [2.2] process captured meas 
if isempty(meas_names)
	meas_names = listdir(fullfile(root,'meas/',meas_dir));
end

% process whole meas
meas_full_save_dir = fullfile(root,'process/full/meas/'); 
mkdir(meas_full_save_dir)

for meas_name = meas_names
	meas_name = char(meas_name);
    disp(['Processing - ',meas_name])
    meas_ori = imread(fullfile(root,'meas/',meas_dir,meas_name));
    meas_debkg = double(meas_ori) - double(black_meas_mean);
    figure(21)
    imshow(uint16(meas_debkg),[])
	meas = meas_debkg;
	
	if save_patch==0
	% [save full measurement]
	save([meas_full_save_dir 'meas_' meas_dir '_' meas_name(1:end-4) '.mat'], 'meas') 
	
	else 
	% [save patched measurement]
		% padding to meas
		meas_debkg_full = zeros(max(img_sz(1),y_max),max(img_sz(2),x_max));
		meas_debkg_full(1:img_sz(1),1:img_sz(2)) = meas_debkg;
		% crop whole meas into patches
		save_dir = fullfile(root,sprintf('process/patch/overlap_%d',overlap));
		save_dir_meas = fullfile(save_dir,'meas',meas_dir,meas_name);
		mkdir(save_dir_meas)
		count = 0;
		figure(22)
		for y = y_min:(ps-overlap):(y_max-ps+1)
			for x = x_min:(ps-overlap):(x_max-ps+1)
				count = count + 1;
				if mod(count,10) == 0
					disp(count)
				end
				meas = meas_debkg_full(y:y+ps-1, x:x+ps-1) / 65535;
				meas_save_path = fullfile(save_dir_meas,sprintf('meas%s.mat',num2str(count,'%03d')));
				save(meas_save_path,'meas')
				imshow(meas*2)
				title(sprintf('meas%d',count))
				pause(0.1)
			end
		end
	end
	disp(['Finished - ', meas_dir, '/',meas_name])
end

%%%%%%%%%%%%%%%%%%%%%%%% appendix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% append -  ROI
% load data
% load('mask_full.mat')
% load('meas_full.mat')

% param setting
root = 'E:\project\SCI_captioning\experiment\realexp\test\SCICAP_outdoor_Cr8_20211213';
save_data_name = 'RecordedImage_GO-5000M-USB__039';
roi = [1860 1010]; % [x,y]
sz = [512 512]; % [w,h]

% processing
save_name = sprintf('roi_test_%s_roi%d-%d_sz%d.mat',save_data_name,roi(1),roi(2),sz(1));
meas_full_save_dir = fullfile(root,'process/full/data/'); 
mkdir(meas_full_save_dir)

mask = mask(roi(2):roi(2)+sz(2)-1,roi(1):roi(1)+sz(1)-1,:);
meas = meas(roi(2):roi(2)+sz(2)-1,roi(1):roi(1)+sz(1)-1);
save([meas_full_save_dir save_name],'meas','mask');


%%%%%%%%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% functions
function [file_names, file_num] = listdir(file_path, file_type)
% get_file_names: get file names and account from given path and filetype
%	input:
%       file_path, like 'E:\'
%       file_type, like '.png'
%   output:
%       file_nums in cell type, file_num
% 
%	Author: Zhihong Zhang, 2021-12-13

if nargin<2
	file_type='';
end

if ~isfolder(file_path)
     warning([file_path 'is not a directory!'])
end
files = dir(fullfile(file_path, ['*', file_type]));
file_names = string({files.name});
file_names(file_names=="."|file_names=="..")=[];
file_num = length(file_names);

end
