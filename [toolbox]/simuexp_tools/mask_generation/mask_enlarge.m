function dst_mask = mask_enlarge(mask, dst_size)
%MASK_ENLARGE enlarge the given mask to get a new mask with almost the same
%balck-white distribution.
%   This function can virtually divide a mask elements into several new elements which has
%   the same value. (i.e. mask resampling to get refined elements)
% 
%   Input:
%   --------
%   - img:      original image, 2D matrix or 3D array([row, col, num])
% 
%   - dst_size: size of destination, 2-element int vector or int scalar
% 
% 
%   Note:
%   --------
%   First, use "kron" to conduct the first enlargement to get a 
%   tmp mask with size approximate to given size. "kron" can only
%   enlarge the mask to a integer times, but it's accurate. 
%   Then use "imresize" to conduct the second enlargement, to 
%   enlarge the tmp mask to the given size. Two steps enlargement
%   can get accurate result than direct "imresize"
%               
% 
%   Output:
%   --------
%   - dst_img: image after average pooling, 2D matrix
% 
%   See also:
%   --------
%   MASK_AVER_POOL
%   
%   Info:
%   --------
%   Created:    Zhihong Zhang <z_zhi_hong@163.com>, 2020-03-29
%   Last Modified:   Zhihong Zhang, 2020-11-25
%
%   Copyright (c) 2020 Zhihong Zhang

% input check
if isvector(dst_size)
     if ~isrow(dst_size)
        dst_size = dst_size(:)';
     end
elseif isscalar(dst_size)
    dst_size = [dst_size dst_size];
else
     error("error input - 'dst_size'");
end


% enlarge
sz_ratio = dst_size./[size(mask,1),size(mask,2)];
nmask = size(mask,3);
dst_mask = zeros([dst_size nmask]);

for	k = 1:nmask
	tmp_mask = kron(mask(:,:,k), ones(floor(sz_ratio)));
	if any(floor(sz_ratio)~=sz_ratio)
		dst_mask(:,:,k) = imresize(tmp_mask, dst_size, 'nearest'); 
	else
		dst_mask(:,:,k) = tmp_mask;
	end
end

end
