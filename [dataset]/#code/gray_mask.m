function mask = gray_mask(sz, mode, param)
%GRAY_MASK	generate a gray mask with given size and other info.
% 
%The gray mask can be generated in a random or determined way.
%It can also generate a gray mask from given image.
% 
%   Input:
%   --------
%   - size: mask size, scalar or int 2D-vector
% 
%   - mode: mask's generating method, str, {'randn', 'rand', 'fixed', 'image'},
%			optional, default='rand'. 'randn' use RANDN to generate a random mask,
%			and 'rand' use RAND to generate a random mask. The other modes generate fixed
%			mask from given request or image.
% 
%   - param:
%           - for 'rand' mode: 'param' is a vector containing [a, b, kernel_size],
%            a,b-float, 0-1; kernel_size-int scalar; a and b is the boundary of
%            the uniform distribution, which determines the average transmittance and variance.
%			And kernel_size determines the average filter kernel size, which affects 
%           the granularity of gray mask, default=[0, 1, 1];
% 
%           - for 'randn' mode: 'param' is a vector containing [mu, sigma,
%           kernel_size], mu-float, [0 1]; sigma-float; kernel_size-int scalar.
%           mu and sigma determine the gaussian random distribution, which
%           affects the average transmittance and variance. And kernel_size
%           determines the average filter kernel size, which affects the granularity 
%           of gray mask, default=[0.5,0.5,1];
% 
%           - for 'fixed' mode: 'param' is the pattern type, str,
%           {'up_half', 'down_half'}
% 
%           - for 'image' mode: 'param' is an image, it will be resized to 
%           given size and then converted to a binary matrix with 'im2bw'
%           
%   Output:
%   --------
%   - mask: gray mask, float 2D matrix,[0 1]
% 
%   Example:
%   --------
%	mask = GRAY_MASK(mask_size); % 0-1 uniform disstribution random mask
%		(equal to: mask = GRAY_MASK(mask_size, 'rand'), 'rand' can be omitted, similarly hereinafter);
% 
%	mask = GRAY_MASK(mask_size, [a, b]); % a-b uniform disstribution 
% 
%	mask = GRAY_MASK(mask_size, [a, b, kernel_size]); % a-b uniform disstribution 
%		random mask, with a average convolution of given kernel_size
% 
% 
%	mask = GRAY_MASK(mask_size, 'randn'); % guassian disstribution mask, 
%		m=0 and sigma=1
% 
%	mask = GRAY_MASK(mask_size, 'randn', [mu, sigma]); % guassian disstribution mask, 
%		mu and sigma are given 
% 
%	mask = GRAY_MASK(mask_size, 'randn', [mu, sigma, kernel_size]); % guassian disstribution mask,
%		mu and sigma are given , with a average convolution of given kernel_size
% 
%   See also:
%   --------
%   BINARY_MASK, RANDN
% 
%   Log:
%   --------
% 
%   Info:
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-03-30
%   Last Modified:  Zhihong Zhang, 2020-05-04
%               
%   Copyright 2020 Zhihong Zhang

% input setting
narginchk(1,3);
if nargin == 1
	% default mode-'rand' with default params
    mode = 'rand';
    param = [0 1 1];
elseif nargin == 2 
    if isnumeric(mode) && isvector(mode)
		% default mode-'rand' with assigned params
        param = mode;
        mode = 'rand';
	elseif strcmp(mode,'rand')
		% default mode-'rand' with default params
		param = [0,1,1];		
	elseif strcmp(mode,'randn')
		% default mode-'randn' with default params
		param = [0.5,0.5,1];
	else
        error('input error!');
    end
end

if isscalar(sz)
    sz = [sz sz];
end
    
% mask generating
switch mode
	case 'rand'
        if isnumeric(param) && isvector(param)
            param_num = numel(param);
			if param_num == 3
                a = param(1); b = param(2); kernel_size = param(3);
            elseif param_num == 2
                a = param(1); b = param(2); kernel_size = 1;
			elseif param_num == 1
				a = 0; b = param(1); kernel_size = 1;
			end
            mask = a + (b-a).*rand(sz); % use uniform random distribution
			if kernel_size~=1
				nchannel = size(mask, 3);
				kernel = fspecial('average',kernel_size);
				if nchannel==1	
					mask = filter2(kernel, mask);
				else
					for k=1:nchannel
						mask(:,:,k) = filter2(kernel, mask(:,:,k));
					end
				end
			end
        else
            error('error input - param')
        end		
    case 'randn'
        if isnumeric(param) && isvector(param)
            param_num = numel(param);
            if param_num == 3
                mu = param(1); sigma = param(2); kernel_size = param(3);
            elseif param_num == 2
                mu = param(1); sigma = param(2); kernel_size = 1;
            elseif param_num == 1
                mu = param(1); sigma = 0.5; kernel_size = 1;
            end
            
            mask = mu + sigma*randn(sz); % use gaussian random distribution
			if kernel_size~=1
				nchannel = size(mask, 3);
				kernel = fspecial('average',kernel_size);
				if nchannel==1	
					mask = filter2(kernel, mask);
				else
					for k=1:nchannel
						mask(:,:,k) = filter2(kernel, mask(:,:,k));
					end
				end
			end
        else
            error('error input - param')
        end
    case 'fixed'
        error('Not implement')
    case 'image'
        if isnumeric(param) && ismatrix(param)
            img = im2double(imresize(param, sz));
        	mask = img/max(img,[],'all'); 
        elseif  isnumeric(param) && (ndims(param)==3 && size(param,3)==3)
            img = im2double(imresize(rgb2gray(param), sz));
            mask = img/max(img,[],'all'); 
        else
            error('error arguement for - "%s"', mode);
        end
end

% clamp to 0-1
mask = max(mask, 0);
mask = min(mask, 1);
end
