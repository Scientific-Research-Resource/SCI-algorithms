function [patch_loc_tmp] = patch_idx_map(img_size, patch_size, skip_size, border)
%PATCH_IDX_MAP: calculate the index map that mapping image to patches (plane mapping) 
%   - border: whether keep borders of the image when it cannot be separated exactly

if(nargin < 4), border = 'off'; end
% vertexs' x and y coordinates of patches (haven't considerd borders)
x = 1:skip_size(1):img_size(1)-patch_size(1)+1;
y = 1:skip_size(2):img_size(2)-patch_size(2)+1;

% dealing with the border patches: if border='on' and the image can't be 
% divided exactly, keep the rest border patches by extend them from border
% to inside to form a complete patch.
if(strcmp(border,'on'))
    if(mod(img_size(1)-patch_size(1), skip_size(1)) ~= 0)
        x = [x img_size(1)-patch_size(1)+1];
    end
    if(mod(img_size(2)-patch_size(2), skip_size(2)) ~= 0)
        y = [y img_size(2)-patch_size(2)+1];
    end
end

% calculate the index map that mapping image to patches
[X,Y] = meshgrid(x,y); % vertexs' x and y coordinates of patches
[dX,dY] = meshgrid(0:patch_size(1)-1,0:patch_size(2)-1); % x and y coordinates of pixels in a patch (based with (0,0))

dX = repmat(reshape(X,[1 1 size(X(:),1)]), [patch_size(1) patch_size(2) 1]) + repmat(dX, [1 1 size(X(:),1)]);
dY = repmat(reshape(Y,[1 1 size(Y(:),1)]), [patch_size(1) patch_size(2) 1]) + repmat(dY, [1 1 size(Y(:),1)]);

patch_loc_tmp = dX+(dY-1)*img_size(1); % sub2ind
patch_loc_tmp = permute(patch_loc_tmp, [2,1,3]);
end
