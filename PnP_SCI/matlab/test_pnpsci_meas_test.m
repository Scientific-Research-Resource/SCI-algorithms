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
% datasetdir = './dataset/simdata/benchmark/binary_mask_256_10f/bm_256_10f'; 
datasetdir = './dataset/simdata/benchmark/combine_binary_mask_256_10f/bm_256_10f'; 
% datasetdir = './dataset/simdata/test/binary_mask_256_10f/bm_rescale_256_10f'; 
% datasetdir = './dataset/simdata/test/combine_binary_mask_512_10f/bm_rescale_512_10f'; 
% datasetdir = './dataset/simdata/cacti/cacti_256_10f_1';
resultdir  = './results';                   % results

saving_data_flag = 0; % save the recon result
tv_init_flag = 0; % use gap-tv recon as initial image for gap-ffdnet

% [1] load dataset
% dataname = 'data_aerial'; % data name
% dataname = 'data_crash'; 
% dataname = 'data_drop'; 
% dataname = 'data_kobe'; 
% dataname = 'data_runner'; 
dataname = 'data_traffic';

% dataname = 'cacti_data_256_10f_1'; 

% dataname = 'traffic_hadamard_8f';

% dataname = 'data_train3_binary'; %  data name
% dataname = 'traffic_8f'; %  data name

datapath = sprintf('%s/%s.mat',datasetdir,dataname);

if exist(datapath,'file')
    load(datapath,'meas','mask','orig'); % meas, mask, orig

	% zzh- normalize
	mask_max = max(mask,[],'a');
	mask = mask./ mask_max;
	meas = meas./ mask_max;

else
    error('File %s does not exist, please check dataset directory!',datapath);
end

nframe = size(meas, 3); % number of coded frames to be reconstructed
nmask  = size(mask, 3); % number of masks (or compression ratio B)
MAXB   = 255;           % maximum pixel value of the image (8-bit -> 255)

para.nframe = nframe; 
para.MAXB   = MAXB;

% [2] apply PnP-SCI for reconstruction
para.Mfunc  = @(z) A_xy(z,mask);
para.Mtfunc = @(z) At_xy_nonorm(z,mask);

para.Phisum = sum(mask.^2,3);
para.Phisum(para.Phisum==0) = 1;

% [2.0] common parameters
mask = single(mask);
orig = single(orig);

para.lambda   =    1; % correction coefficiency
para.acc      =    1; % enable acceleration
para.flag_iqa = true; % enable image quality assessments in iterations


% [2.1] GAP-TV
para.denoiser = 'tv'; % TV denoising
para.maxiter  = 100; % maximum iteration
% para.tvweight = 0.07*255/MAXB; % weight for TV denoising, original
% para.tviter   = 5; % number of iteration for TV denoising, original

para.tvweight = 0.07*255/MAXB; % weight for TV denoising, test
para.tviter   = 5; % number of iteration for TV denoising, test

[vgaptv,psnr_gaptv,ssim_gaptv,tgaptv,psnrall_tv] = ...
    gapdenoise_cacti(mask,meas,orig,[],para);

fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gaptv),mean(ssim_gaptv),tgaptv);


% [2.2] GAP-FFDNet
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
		gapdenoise_cacti(mask,meas,orig,vgaptv,para); 
else
	[vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,tgapffdnet,psnrall_ffdnet] = ...
	    gapdenoise_cacti(mask,meas,orig,[],para);
end


fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gapffdnet),mean(ssim_gapffdnet),tgapffdnet);


% [3] save as the result .mat file 
if saving_data_flag
	matdir = [resultdir '/savedmat'];
	if ~exist(matdir,'dir')
		mkdir(matdir);
	end

	% save([matdir '/pnpsci_' dataname num2str(nframe*nmask) '.mat']);
	% zzh
	if ~exist([matdir '/pnpsci_' dataname num2str(nframe*nmask) '.mat'], 'file')
		save([matdir '/pnpsci_' dataname num2str(nframe*nmask) '.mat']);
	else
		save([matdir '/pnpsci_' dataname num2str(nframe*nmask) '.mat'], '-append');
	end
end
