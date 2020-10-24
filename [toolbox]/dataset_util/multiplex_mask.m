function new_mask = multiplex_mask(init_mask, multiplex_matrix)
%MULTIPLEX_MASK linearly multiplex given masks to get new masks 
% 
%   Input:
%   --------
%   - init_mask: original masks, 3D array, numerical, [nrow, ncol, nmask]
% 
%   - multiplex_matrix: indication matrix for multiplexing, 2D matrix, [n_newmask nmask]. 
%	  each row of the multiplex_matrix indicates a new mask, the row
%	  elements are the linear combination coefficients of corresponding
%	  init_mask.
% 
%   Note£º
%   --------
%	new_mask(i) = sum(j=1:nmask)(multiplex_matrix(i,j)*init_mask(:,:,j))
% 
%   Info£º
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-04-26
%   Last Modified:  Zhihong Zhang <z_zhi_hong@163.com>, 2020-04-26
%   
%   Copyright 2020 Zhihong Zhang

mask_size = size(init_mask);
multiplex_res_num = size(multiplex_matrix,1);

tmp_mask = reshape(init_mask, [prod(mask_size(1:2)), mask_size(3)]);
tmp_mask = tmp_mask'; %size=[mask_size(3) prod(img_size)]

mask = multiplex_matrix*tmp_mask;
mask = mask';
new_mask = reshape(mask, [mask_size(1:2), multiplex_res_num]);

end