function [v, psnrall] = gap_joint_denoise( y , opt )
%GAP_JOINT_DENOISE Generalized alternating projection (GAP)-based joint denoising
%framework for compressive sensing reconstruction.
%   v=GAPDENOISE(y,opt) returns the reconstruction result v of the
%   measurements with CASSI or CACTI coding, where y is the measurement
%   matrix, opt is the parameters for the GAP-Denoise algorithm, typically
%   the denoiser applied in the framework.
% Reference
%   [1] Y. Liu, X. Yuan, J. Suo, D.J. Brady, and Q. Dai, Rank Minimization 
%       for Snapshot Compressive Imaging, IEEE Trans. Pattern Anal. Mach. 
%       Intell. (TPAMI), DOI:10.1109/TPAMI.2018.2873587, 2018.
%   [2] X. Yuan, Generalized alternating projection based total variation 
%       minimization for compressive sensing, in Proc. IEEE Int. Conf. 
%       Image Process. (ICIP), pp. 2539-2543, 2016.
% Code credit
%   Xin Yuan, Bell Labs, xyuan@bell-labs.com, initial version Jul 2, 2015.
%   Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, last
%     update Jul 13, 2018.
% 
%   See also GAPDENOISE_CACTI.
if nargin<2
    opt = [];
end
addpath(genpath('./tvdenoisers')); % tvdenoisers

% [0] initialized v0
At = @(z) Mt_func(z);
if isfield(opt,'Mtfunc'), At = @(z) opt.Mtfunc(z); end             
if isfield(opt,'v0'),             v0 = opt.v0;       end
if ~exist('v0','var') || isempty(v0)
    v0 = At(y); % start point (initialization of iteration)
end

v = v0; % initialization

% [1] 1st period denoising: gaptv denoising
disp('*** 1st period denoising ***')
[v, ~] = gap_tv_denoise(y, v, opt);

% [2] 2nd period denoising: gap tv+ffdnet denoising
disp('*** 2nd period denoising ***')
[v, psnrall] = gap_multistep_denoise(y, v, opt);

end            
            
%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function: gap_tv_denoise
function [denoised_v, psnrall] = gap_tv_denoise(y, v, opt)
	disp('--- do gap_tv_denoise --- ');
	% -- params --
	denoiser = 'tv';  % video denoiser
	lambda   = 0.2;     % correction coefficiency
	% maxiter  = 100;     % maximum number of iteration
	% sigma    = 10/255;  % noise deviation 
	acc      = 1;       % enable acceleration
	tvweight = 0.07;    % weight for TV denoising
	tviter   = 5;       % number of iteration for TV denoising
	tvm = 'ATV_ClipA';	% tv denoiser
	% nosestim = true;    % enable noise estimation (if possible)
	flag_iqa = true;    % flag of showing image quality assessments (be sure 
						%  to turn it off for benchmark)
	% ffdnetvnorm_init = true; % use normalized video as input for initialization
							 %  (with the first 10 iterations)	
							 
	A  = @(x) M_func(x);
	At = @(z) Mt_func(z);
	if isfield(opt,'Mfunc'),  A  = @(x) opt.Mfunc(x);  end
	if isfield(opt,'Mtfunc'), At = @(z) opt.Mtfunc(z); end							 
	if isfield(opt,'Phisum'),     Phisum = opt.Phisum;   end
	% if isfield(opt,'denoiser'), denoiser = opt.denoiser; end
	if isfield(opt,'tvm'),			 tvm = opt.tvm;		 end
	% if isfield(opt,'v0'),             v0 = opt.v0;       end
	if isfield(opt,'lambda'),   lambda   = opt.lambda;   end
	% if isfield(opt,'maxiter'),   maxiter = opt.maxiter;  end
	if isfield(opt,'acc'),           acc = opt.acc;      end
	if isfield(opt,'tvweight'), tvweight = opt.tvweight; end
	if isfield(opt,'tviter'),     tviter = opt.tviter;   end
	% if isfield(opt,'nosestim'), nosestim = opt.nosestim; end
	% if isfield(opt,'sigma'),       sigma = opt.sigma;    end
	if isfield(opt,'flag_iqa'), flag_iqa = opt.flag_iqa; end
	if isfield(opt,'ffdnetvnorm_init'), ffdnetvnorm_init = opt.ffdnetvnorm_init; end

	y1 = zeros(size(y),'like',y);
	k = 1;
	psnrall = []; % return empty with no ground truth	
	
	% --  processing --
	for iter = 1:tviter
% 		if (mod(iter, 10) == 1)
% 			if flag_iqa && isfield(opt,'orig') && (~isempty(opt.orig))
% 				%PSNR     =   psnr( theta, para.ori_im);      
% 				PSNR_f     =   psnr(double(v),double(opt.orig));      
% 				disp( [tvm ' GAP Image Recovery, Iter ' num2str(iter) ':, PSNR v = ' num2str(PSNR_f)]); %, iter, PSNR, PSNR_f] );
% 			end
% 		end		
		
		yb = A(v);
		if acc % enable acceleration
			y1 = y1+(y-yb);
			v = v+lambda*(At((y1-yb)./Phisum)); % v=v+lambda*(At*A)^-1*At*dy
		else
			v = v+lambda*(At((y-yb)./Phisum));
		end			
		switch tvm
			case 'ATV_ClipA'
				v         =   TV_denoising(v,  para.tvweight,5);  
			case 'ATV_ClipB'
				v         =    TV_denoising_clip_LB(v,  tvweight,5); 
			   %  f         =    TV_denoising_ClipB(f,  para.tvweight,2e-2,50,5); 
			case 'ATV_cham'
				v         =     tvdenoise_cham_ATV2D(v,  1/tvweight,5);  
			case 'ATV_FGP'
				v         =     fgp_denoise_ATV2D(v, tvweight,2);  
			case 'ITV2D_cham'
				v         =     tvdenoise_cham_ITV2D(v,  1/tvweight,5);  
			case 'ITV2D_FGP'
				v         =     fgp_denoise_ITV2D(v,  tvweight,2);  
			case 'ITV3D_cham'
				v         =     tvdenoise_cham_ITV3D(v,  1/tvweight,5);  
			case 'ITV3D_FGP'
				v         =     fgp_denoise_ITV3D(v,  tvweight,2);  
		end
		
		% show intermediate results of psnr (and ssim)
		if flag_iqa && isfield(opt,'orig') && (~isempty(opt.orig))
			psnrall(k) = psnr(double(v),double(opt.orig)); % record all psnr
			if (mod(k,5)==0) 
				fprintf('  GAP-%s-%s iteration % 4d, sigma -, PSNR %2.2f dB.\n',...
					upper(denoiser),upper(tvm),k,psnrall(k));
			end
		end
		k = k+1;
		
	end
	
	denoised_v = v;
end


%% function: gap_multistep_denoise
function [denoised_v, psnrall] = gap_multistep_denoise(y, v, opt)
	disp('--- do gap_multistep_denoise --- ');
	% --  params --
	denoiser = 'tv';  % video denoiser
	lambda   = 0.2;     % correction coefficiency
	maxiter  = 100;     % maximum number of iteration
	sigma    = 10/255;  % noise deviation 
	acc      = 1;       % enable acceleration
	% tvweight = 0.07;    % weight for TV denoising
	intviter   = 5;       % number of iteration for TV denoising
	intvm = 'ATV_ClipA';	% ineer tv denoiser (multi-denoise situation)
	% nosestim = true;    % enable noise estimation (if possible)
	flag_iqa = true;    % flag of showing image quality assessments (be sure 
						%  to turn it off for benchmark)
	ffdnetvnorm_init = true; % use normalized video as input for initialization
							 %  (with the first 10 iterations)	

	A  = @(x) M_func(x);
	At = @(z) Mt_func(z);
	if isfield(opt,'Mfunc'),  A  = @(x) opt.Mfunc(x);  end
	if isfield(opt,'Mtfunc'), At = @(z) opt.Mtfunc(z); end
	if isfield(opt,'Phisum'),     Phisum = opt.Phisum;   end
	if isfield(opt,'denoiser'), denoiser = opt.denoiser; end
	% if isfield(opt,'tvm'),			 tvm = opt.tvm;		 end
	% if isfield(opt,'v0'),             v0 = opt.v0;       end
	if isfield(opt,'lambda'),   lambda   = opt.lambda;   end
	if isfield(opt,'maxiter'),   maxiter = opt.maxiter;  end
	if isfield(opt,'acc'),           acc = opt.acc;      end
	if isfield(opt,'tvweight'), tvweight = opt.tvweight; end
	if isfield(opt,'intvm'),	   intvm = opt.intvm;		 end
	if isfield(opt,'intviter'),     intviter = opt.intviter;   end
	% if isfield(opt,'nosestim'), nosestim = opt.nosestim; end
	if isfield(opt,'sigma'),       sigma = opt.sigma;    end
	if isfield(opt,'flag_iqa'), flag_iqa = opt.flag_iqa; end
	if isfield(opt,'ffdnetvnorm_init'), ffdnetvnorm_init = opt.ffdnetvnorm_init; end

	if  isfield(opt,'ffdnetvnorm') && ffdnetvnorm_init
		sigma = [50/255 sigma];
		maxiter = [10 maxiter];
		ffdnetvnorm = opt.ffdnetvnorm;
	end

	y1 = zeros(size(y),'like',y);

	% -- processing --
	k = 1; % current number of iteration
	psnrall = []; % return empty with no ground truth	
	
	for isig = 1:length(maxiter) % extension for a series of noise levels
		nsigma = sigma(isig); 
		opt.sigma = nsigma;
		for iter = 1:maxiter(isig)
			% [1.1] Euclidean projection
			yb = A(v);
			if acc % enable acceleration
				y1 = y1+(y-yb);
				v = v+lambda*(At((y1-yb)./Phisum)); % v=v+lambda*(At*A)^-1*At*dy
			else
				v = v+lambda*(At((y-yb)./Phisum));
			end

			% [1.2] Denoising to match the video prior
			switch lower(denoiser)
				case 'tv+ffdnet' % tv + ffdnet joint denoising
					if isig==1 && iter==1
						disp(' ...tv+ffdnet_denoising...');
					end
					% [.1] denoise_step1: tv denoising
					% v = TV_denoising(v,tvweight,intviter);
					v = tvdenoise_cham_ITV2D(v,  1/tvweight,intviter); 
					% v = fgp_denoise_ITV3D(v,  tvweight,2);
					switch intvm
						case 'ATV_ClipA'
							v         =   TV_denoising(v,  para.tvweight,5);  
						case 'ATV_ClipB'
							v         =    TV_denoising_clip_LB(v,  tvweight,5); 
						   %  f         =    TV_denoising_ClipB(f,  para.tvweight,2e-2,50,5); 
						case 'ATV_cham'
							v         =     tvdenoise_cham_ATV2D(v,  1/tvweight,5);  
						case 'ATV_FGP'
							v         =     fgp_denoise_ATV2D(v, tvweight,2);  
						case 'ITV2D_cham'
							v         =     tvdenoise_cham_ITV2D(v,  1/tvweight,5);  
						case 'ITV2D_FGP'
							v         =     fgp_denoise_ITV2D(v,  tvweight,2);  
						case 'ITV3D_cham'
							v         =     tvdenoise_cham_ITV3D(v,  1/tvweight,5);  
						case 'ITV3D_FGP'
							v         =     fgp_denoise_ITV3D(v,  tvweight,2);  
					end					
					
					% [.2] denoise_step2: ffdnet_denoising
					if ffdnetvnorm_init
						if isig==1
							opt.ffdnetvnorm = true;
						else
							opt.ffdnetvnorm = ffdnetvnorm;
						end
					end
					v = ffdnet_vdenoise(v,[],opt); % opt.sigma
				otherwise
					error('Unsupported denoiser %s!',denoiser);
			end

			% [1.3] save and show intermediate results of psnr (and ssim)
			if flag_iqa && isfield(opt,'orig') && (~isempty(opt.orig))
				psnrall(k) = psnr(double(v),double(opt.orig)); % record all psnr
				if (mod(k,5)==0) 
					fprintf('  GAP-%s iteration % 4d, sigma % 3d, PSNR %2.2f dB.\n',...
						upper(denoiser),k,nsigma*255,psnrall(k));
				end
			end
			k = k+1;
		end % GAP loop [maxiter]
	end % sigma loop [length(maxiter)]
	
	denoised_v = v;
end