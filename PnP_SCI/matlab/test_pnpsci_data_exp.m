% function test_pnpsci(dataname)
%TEST_PNPSCI Test Plug-and-Play algorithms for Snapshot Compressive Imaging
%(PnP-SCI). Here we include the deep denoiser FFDNet as image/video priors 
%along with TV denoiser/prior for six simulated benchmark video-SCI datasets 
%(in grayscale).
%   TEST_PNPSCI(dataname) runs the PnP-SCI algorithms with TV and FFDNet as
%   denoising priors for grayscale video SCI, where dataname denotes the
%   name of the dataset for reconstruction with `kobe` data as default.
% Reference
%   [1] X. Yuan, Y. Liu, J. Suo, and Q. Dai, Plug-and-play Algorithms for  
%       Large-scale Snapshot Compressive Imaging, in IEEE/CVF Conf. Comput. 
%       Vis. Pattern Recognit. (CVPR), 2020.
%   [2] X. Yuan, Generalized alternating projection based total variation 
%       minimization for compressive sensing, in Proc. IEEE Int. Conf. 
%       Image Process. (ICIP), pp. 2539-2543, 2016.
% Dataset
%   Please refer to the readme file in `dataset` folder.
% Contact
%   Xin Yuan, Bell Labs, xyuan@bell-labs.com, initial version Jul 2, 2015.
%   Yang Liu, MIT CSAIL, yliu@csail.mit.edu, last update Apr 1, 2020.
%   
%   See also GAPDENOISE_CACTI, GAPDENOISE, TEST_PNPSCI_LARGESCALE.

% [0] environment configuration
clc, clear
addpath(genpath('./algorithms')); % algorithms
addpath(genpath('./packages'));   % packages
addpath(genpath('./utils'));      % utilities

% datasetdir = './dataset/simdata/benchmark'; % benchmark simulation dataset
% datasetdir = './dataset/simdata/test_data';  % dataset for test

% datasetdir = './dataset/simdata/benchmark/data/binary_mask_256_10f/bm_256_10f'; 
% datasetdir = 'E:\project\CACTI\experiment\real_data\data_cacti\#mask_bin4_Cr10_20210107\_dataset'; 
% datasetdir = 'E:\project\CACTI\experiment\real_data\data_cacti\#mask_bin4_20210126\scene\d_alpha';
% datasetdir = 'E:\project\CACTI\experiment\real_data\data_cacti\#mask_bin4_20210126\scene\d_hand';
datasetdir = 'E:\project\CACTI\experiment\real_data\data\#mask_side_circ_20210312\scene';

result_dir  = './results';                   % results

test_algo_flag = ["admmtv","istatv","gaptv+ffdnet"];		% choose algorithms: 0-all, 1-gaptv, 2-gapffdnet, 3-istatv, 4-gaptv+ffdnet,5-admmtv
saving_data_flag = 1;	% save the recon result
tv_init_flag = 0;		% use gap-tv recon as initial image for gap-ffdnet
show_res_flag = 1;

% [1] load dataset
% dataname = 'data_aerial'; % data name
% dataname = 'data_d_alpha2_20201209_roi820-615_sz512_dataBinned4'; 
% dataname = 'data_d_hand1_20210126_roi970-350_sz512_dataBinned4';
% dataname = 'data_d_hand1_20210126_roi690-180_sz1024_dataBinned2';
dataname = 'data_opto (3)_20210312_roi1165-1645_sz400';  %_dataBinned2

datapath = sprintf('%s/%s.mat',datasetdir,dataname);

if exist(datapath,'file')
%     load(datapath,'meas','mask','orig'); % meas, mask, orig
	data = load(datapath); % meas, mask
	mask = data.mask;
	meas = data.meas;	
	
	orig=[];
	
	% zzh- normalize 
	mask_max = max(mask,[],'a');
	mask = mask./ mask_max;
% 	mask= single(imbinarize(mask));
	meas = meas./ mask_max;

else
    error('File %s does not exist, please check dataset directory!',datapath);
end

nframe = size(meas, 3); % number of coded frames to be reconstructed
nmask  = size(mask, 3); % number of masks (or compression ratio B)
% MAXB   = 255;           % maximum pixel value of the image (8-bit -> 255)
MAXB   = 65535;					% maximum pixel value of the image (16-bit -> 65535), for simulated data
% MAXB   = 65535/nmask;           % maximum pixel value of the image (16-bit -> 65535), for rel data

para.nframe = nframe;	
para.MAXB   = MAXB;

% [2] apply PnP-SCI for reconstruction
para.Mfunc  = @(z) A_xy(z,mask);
para.Mtfunc = @(z) At_xy_nonorm(z,mask);
para.gradF  = @(z) gradf(z, meas/MAXB, mask); % zzh: for ista

para.Phisum = sum(mask.^2,3);
para.Phisum(para.Phisum==0) = 1;

% [2.0] common parameters
mask = single(mask);
orig = single(orig);

para.lambda   =    1; % correction coefficiency
para.acc      =    1; % enable acceleration
para.flag_iqa = true; % enable image quality assessments in iterations

% [2.1] GAP-TV
if ismember("gaptv",test_algo_flag) || ismember("all",test_algo_flag)
	para.denoiser = 'tv'; % TV denoising
% 	para.tvm = 'ITV3D_FGP';  % tv denoiser
	para.tvm = 'ATV_ClipA';  % tv denoiser
	para.maxiter  = 100; % maximum iteration
	% para.tvweight = 0.07*255/MAXB; % weight for TV denoising, original
	% para.tviter   = 5; % number of iteration for TV denoising, original

	para.tvweight = 0.15*255/MAXB; % weight for TV denoising, test
	para.tviter   = 5; % number of iteration for TV denoising, test

	[vgaptv,psnr_gaptv,ssim_gaptv,tgaptv,psnrall_gaptv] = ...
		gapdenoise_cacti(mask,meas,orig,[],para);

	fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
		upper(para.denoiser),mean(psnr_gaptv),mean(ssim_gaptv),tgaptv);
	disp('===== GAP-TV Finished! =====')
end

% [2.2] GAP-FFDNet
if ismember("gapffdnet",test_algo_flag) || ismember("all",test_algo_flag)
	para.denoiser = 'ffdnet'; % FFDNet denoising
	load(fullfile('models','FFDNet_gray.mat'),'net');

	para.net = vl_simplenn_tidy(net);
	para.useGPU = true;
	if para.useGPU
	  para.net = vl_simplenn_move(para.net, 'gpu') ;
	end
	para.ffdnetvnorm_init = true; % use normalized video for the first 10 iterations
	para.ffdnetvnorm = false; % normalize the video before FFDNet video denoising

	% para.sigma   = [50 25 12  6]/MAXB; % default, for kobe
	% para.maxiter = [10 10 10 10]; % default, for kobe
	  para.sigma   = [35 15 12  6]/MAXB; %   for test_kobe_binary
	  para.maxiter = [10 10 10 10];
	% para.sigma   = [12 6]/MAXB; %   for test
	% para.maxiter = [10 10 ];

	if tv_init_flag
		% use gap-tv result as the initialized input
		[vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,tgapffdnet,psnrall_ffdnet] = ...
			gapdenoise_cacti(mask,meas,orig,[],para); 
	else
		[vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,tgapffdnet,psnrall_ffdnet] = ...
			gapdenoise_cacti(mask,meas,orig,[],para);
	end
	
	fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
		upper(para.denoiser),mean(psnr_gapffdnet),mean(ssim_gapffdnet),tgapffdnet);
	disp('===== GAP-FFDNet Finished! =====')
end						  


% [2.3] ISTA-TV
if ismember("istatv",test_algo_flag) || ismember("all",test_algo_flag)
	% denoiser
 	para.denoiser = 'tv'; % TV denoising
% 	para.denoiser = 'ffdnet';
% 	para.denoiser = 'vbm4d';

	if strcmp(para.denoiser, 'tv')
		% tv params
		para.tvweight = 0.15*255/MAXB; % weight for TV denoising, test
		para.tviter   = 5; % number of iteration for TV denoising, test
		para.maxiter  = 200; % maximum iteration
		para.lambda  = 0.3; % regularization factor
	elseif(strcmp(para.denoiser, 'ffdnet'))
		% ffdnet params
		load(fullfile('models','FFDNet_gray.mat'),'net');

		para.net = vl_simplenn_tidy(net);
		para.useGPU = true;
		if para.useGPU
		  para.net = vl_simplenn_move(para.net, 'gpu') ;
		end
		para.ffdnetvnorm_init = true; % use normalized video for the first 10 iterations
		para.ffdnetvnorm = false; % normalize the video before FFDNet video denoising
% 		para.sigma   = [50 25 12  6]/MAXB; % default, for kobe
% 		para.maxiter = [30 20 10 10]; % default, for kobe
		para.sigma   = [20 10 8 6]/MAXB; % default, for kobe
		para.maxiter = [30 20 20 15]; % default, for kobe
		para.lambda  = 0.2; % regularization facto
	elseif(strcmp(para.denoiser, 'vbm4d'))
		para.maxiter = 170; % maximum iteration
		para.lambda  = 0.2; % regularization facto
	end
	
	
	[vistatv,psnr_istatv,ssim_istatv,tistatv,psnrall_istatv] = ...
		istadenoise_cacti(mask,meas,orig,[],para);

	fprintf('ISTA-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
		upper(para.denoiser),mean(psnr_istatv),mean(ssim_istatv),tistatv);
	disp('===== ISTA-TV Finished! =====')
end

% [2.4] GAP-JOINT
if ismember("gaptv+ffdnet",test_algo_flag) || ismember("all",test_algo_flag)
	
	para.denoiser = 'tv+ffdnet'; % FFDNet denoising
	para.tvm = 'ITV3D_FGP';  % tv denoiser
	para.intvm = 'ITV2D_cham';  % ineer tv denoiser (multi-denoise situation)
	load(fullfile('models','FFDNet_gray.mat'),'net');

	para.net = vl_simplenn_tidy(net);
	para.useGPU = true;
	if para.useGPU
	  para.net = vl_simplenn_move(para.net, 'gpu') ;
	end
	para.ffdnetvnorm_init = true; % use normalized video for the first 10 iterations
	para.ffdnetvnorm = false; % normalize the video before FFDNet video denoising

	para.tviter =100;   % 1st period gaptv iteration
	para.intviter = 5;  % inner gaptv iteration
	para.mu = 0.25;
	para.iter =150;
	para.tvweight = 0.05;
	para.tvm = 'ITV3D_FGP';
	 
	para.sigma   = [50 20 10 6]./MAXB;  %   for test_kobe_binary
	para.maxiter = [20 40 100 200];

	
	[vgapjoint,psnr_gapjoint,ssim_gapjoint,tgapjoint,psnrall_joint] = ...
		gap_joint_denoise_cacti(mask,meas,orig,[],para);
	
	fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
		upper(para.denoiser),mean(psnr_gapjoint),mean(ssim_gapjoint),tgapjoint);
	disp('===== GAP-JOINT Finished! =====')
end

% [2.5] ADMM-TV
if ismember("admmtv",test_algo_flag) || ismember("all",test_algo_flag)
	para.denoiser = 'tv'; % TV denoising
	para.tvm = 'ITV3D_FGP';  % tv denoiser
% 	para.tvm = 'ATV_FGP';  % tv denoiser
% 	para.tvm = 'ATV_ClipA';  % tv denoiser
	para.maxiter  = 200; % maximum iteration
	% para.tvweight = 0.07*255/MAXB; % weight for TV denoising, original
	% para.tviter   = 5; % number of iteration for TV denoising, original
 	% param.gamma = 0.2; % gamma for admm (larger noise, larger gamma), original
	param.gamma = 200; % gamma for admm (larger noise, larger gamma)
	para.tvweight = 200; % weight for TV denoising, test
	para.tviter   = 10; % number of iteration for TV denoising, test

	[vadmmtv,psnr_admmtv,ssim_admmtv,tadmmtv,psnrall_admmtv] = ...
		admmdenoise_cacti(mask,meas,orig,[],para);

	fprintf('ADMM-%s-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
		upper(para.denoiser),upper(para.tvm),mean(psnr_admmtv),mean(ssim_admmtv),tadmmtv);
	disp('===== ADMM-TV Finished! =====')
end

% [3] save as the result .mat file 
if saving_data_flag
	matdir = [result_dir '/savedmat'];
	if ~exist(matdir,'dir')
		mkdir(matdir);
	end

	% save([matdir '/pnpsci_' dataname num2str(nframe*nmask) '.mat']);
	% zzh
	if ~exist([matdir '/pnpsci_' dataname '_'  num2str(nframe*nmask) '.mat'], 'file')
		save([matdir '/pnpsci_' dataname '_' num2str(nframe*nmask) '.mat']);
	else
		save([matdir '/pnpsci_' dataname '_' num2str(nframe*nmask) '.mat'], '-append');
	end
end

% [4] show  result
if show_res_flag
	plot_row=2; plot_col=5;
	switch test_algo_flag
		case "gaptv"
			result = vgaptv;
		case "gapffdnet"
			result = vgapffdnet;
		case "istatv"
			result = vistatv;
		case "gaptv+ffdnet"
			result = vgapjoint;
		case "admmtv"
			result = vadmmtv;
		case "all"
			disp("MORE THAN ONE ALGORITHM CHOSEN - please manually check the results")
			return			
	end
% 	result = vgapjoint;
	figure; 
	for tt=1:size(result,3)
		 %theta_gap_tv_rgb0(:,:,:,tt) = im;
		subplot(plot_row, plot_col,tt); imagesc(result(:,:,tt)); colormap gray; title(['Frame: ' num2str(tt)])
	end

	figure;
	for tt=1:size(result,3)
		im = result(:,:,tt);
		imshow(im./max(im(:))); title(tt); pause(0.2); 
	end
end