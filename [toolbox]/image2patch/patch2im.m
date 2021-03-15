function img_recon = patch2im(patches, img_size, skip_size, border)
%PATCH2IM: restore images from patches
%   - patches: small patches, [h, w, c] for gray image; [h, w, c, p] for multi-channel
%	- img_size: the size of original image, 2D int
%	- skip_size: the stride of patches, 2D int
%   - border: whether keep borders of the image when it cannot be separated
%				exactly, string array, value={'on', 'off'}
%	- img_recon: reconstructed image, gray or multi-channel
% 
%	Note: if 'border' is 'on', then the last sub-image will be the "first"
%	sub-image counting from the end of the row or the column.
if(nargin < 4), border = 'off'; end

patch_size = [size(patches,1) size(patches,2)];
img_size_ch1 = img_size(1:2); % reconstruction image size(single channel)

if length(img_size)<3
    channel_num = 1;
else
    channel_num = img_size(3);
end

% restore image from patcher
img_recon = zeros(img_size);

for ch = 1:channel_num
    img_tmp = zeros(img_size_ch1);
    w = zeros(img_size_ch1);
    
    if channel_num == 1
        patches_tmp = patches;
    else
        patches_tmp = squeeze(patches(:,:,ch,:));
    end
    
    % get the index map that mapping image to patches (plane mapping) 
    patch_loc = patch_idx_map(img_size_ch1, patch_size, skip_size, border);
    
    patches_tmp = reshape(patches_tmp, [prod(patch_size) size(patches_tmp,3)]);
    
    for n=1:size(patch_loc,3)
        img_tmp(patch_loc(:,:,n)) = img_tmp(patch_loc(:,:,n)) + reshape(patches_tmp(:,n), patch_size);
        w(patch_loc(:,:,n)) = w(patch_loc(:,:,n)) + 1;
    end
    img_tmp = img_tmp ./ w; %去重复累加的部分
   
    
    if channel_num == 1
        img_recon = img_tmp;
    else
        img_recon(:,:,ch) = img_tmp;
    end
end

% crop the nan area
if strcmp(border, 'off')
    img_x = find(isnan(img_recon(:,:,1)),1);
    img_y = find(isnan(img_recon(:,:,1)'),1);
	if ~isempty(img_x) && ~isempty(img_y)
		img_recon = img_recon(1:img_x-1, 1:img_y-1,:);
	end

end