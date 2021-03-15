function patches = im2patch(src_img, patch_size, skip_size, border)
%IM2PATCH: converts images to patches
%   - src_img: original image, gray or multi-channel
%   - patch_size: size of small patches, 2D int
%	- skip_size: the stride of patches, 2D int
%   - border: whether keep borders of the image when it cannot be separated
%				exactly, string array, value={'on', 'off'}
%   - patches: small patches, [h, w, c] for gray image; [h, w, c, p] for multi-channel
% 
%	Note: if 'border' is 'on', then the last sub-image will be the "first"
%	sub-image counting from the end of the row or the column.

if(nargin < 4), border = 'off'; end

% get the index map that mapping image to patches (plane mapping) 
patch_loc = patch_idx_map(size(src_img), patch_size, skip_size, border);

channel_num = size(src_img, 3);
patch_num=size(patch_loc, ndims(patch_loc));

% get patches from each channnel and merge them together
patches = zeros([patch_size, channel_num, patch_num]);
if channel_num > 1
    for ch = 1:channel_num
        img_ch = src_img(:,:,ch);
        patches(:,:,ch,:) = img_ch(patch_loc);  
    end
else
    patches = src_img(patch_loc); % gray image
end
end