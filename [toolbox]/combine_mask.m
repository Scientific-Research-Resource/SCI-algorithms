function new_mask = combine_mask(init_mask, combine_matrix)
%COMBINE_MASK linearly combine given masks to get new masks 
% 
%   Input:
%   --------
%   - init_mask: original masks, 3D array, numerical, [nrow, ncol, nmask]
% 
%   - combine_matrix: indication matrix for combination, 2D matrix, [n_newmask nmask]. 
%	  each row of the combine_matrix indicates a new mask, the row
%	  elements are the linear combination coefficients of corresponding
%	  init_mask.
% 
%   Note£º
%   --------
%	new_mask(i) = sum(j=1:nmask)(combine_matrix(i,j)*init_mask(:,:,j))
% 
%   Info£º
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-04-26
%   Last Modified:  Zhihong Zhang <z_zhi_hong@163.com>, 2020-04-26
%   
%   Copyright 2020 Zhihong Zhang

img_size = size(init_mask);
img_size = img_size(1:2);
img_num = size(init_mask, 3);

tmp_mask = reshape(init_mask, [prod(img_size), img_num]);
tmp_mask = tmp_mask'; %size=[img_num prod(img_size)]

mask = combine_matrix*tmp_mask;
mask = mask';
new_mask = reshape(mask, [img_size, img_num]);

end