function mask = shift_mask(src_mask, sz, sft_idx, sft_idx_type, center_orig)
%SHIFT_MASK	generate a shifting mask with given source mask and other info.
% 
% 
%   Input:
%   --------
% 
%   - src_mask: the source mask, 2D matrix
% 
%   - sz: shifting mask's size, scalar or int 2D-vector
% 
%   - sft_idx: 
%		sft_idx_type=='coord': shifting coordinates relative to the center mask, 
%		Nx2 array; 
%		sft_idx_type=='range': maximum shifting pixel num, 1x2 array
% 
%	- sft_idx_type: 
% 
%	- center_orig: origin coordinates of the center mask, int, 2D-vector,
%	optional, default = $center of the src_mask
%           
%   Output:
%   --------
%   - mask: shifting mask, 2D matrix
% 
%   See also:
%   --------
%   BINARY_MASK, GRAY_MASK
% 
%   Log:
%   --------
% 
%   Info:
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-10-21
%   Last Modified:  Zhihong Zhang, 2020-10-21
%               
%   Copyright 2020 Zhihong Zhang

% input setting
narginchk(4,5);
if nargin < 5
	% center mask's left-up corner coord (default: center mask at the center of src_mask)
	center_lu = round((size(src_mask)-sz)./2)+1; 
else
	center_lu = center_orig - round(sz./2);
end

if isscalar(sz)
    sz = [sz sz];
end

if strcmp(sft_idx_type,'range')
	blk_sz = sft_idx;

	if isscalar(blk_sz)
		blk_sz = [blk_sz blk_sz];
	end

	if mod(blk_sz(1),2)==0
		sft_x = [-blk_sz(1)/2:-1 1:blk_sz(1)/2];
	else
		sft_x = [1:blk_sz(1)] - ceil(blk_sz(1)/2);
	end

	if mod(blk_sz(2),2)==0
		sft_y = [-blk_sz(2)/2:-1 1:blk_sz(2)/2];
	else
		sft_y = [1:blk_sz(2)] - ceil(blk_sz(2)/2);
	end
	[sft_xx, sft_yy] = meshgrid(sft_x, sft_y);
	sft_array = [sft_xx(:) sft_yy(:)];
elseif strcmp(sft_idx_type,'coord')
	sft_array = sft_idx;
else
	error('sft_idx_type=="coord or "range"')
end

if ~any(size(sft_array)==2)
	error('sft_array - should be a 2D-array')
end

if size(sft_array,2)>2
	sft_array = sft_array';
end

% mask generating
mask_num = size(sft_array,1);
mask = zeros([sz, mask_num]);
sft_lu = center_lu + sft_array; % shifting masks' left-up coords
if min(sft_lu,[], 'all')==0 || any(max(sft_lu)+sz-1>size(src_mask))
	error('shifting out of source mask')
end

for k = 1:mask_num
	mask(:,:,k) = imcrop(src_mask, [sft_lu(k, 2) sft_lu(k, 1) sz(2)-1 sz(1)-1]);
end


end
