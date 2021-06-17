function im= patches2image(M_patches, N1, N2, delta1, delta2, border)
% Reput patches in images in the appropriate positions with appropriate
% weight (average over overlapping areas). It is the inverse procedure of image2patches3d
% M_patches is 3D with size patch x patch x num if patches
% 
% Input
% M_patches:		patches,  [n1, n2, patch_num]
% [N1 N2]:			original image's size
% [delta1 delta2]:	the step size
% border:			whether keep borders of the image when it cannot be
%					separated exactly, 'on' | 'off', default='on'
%
% Output
% im:				original image
% Xin Yuan, 2015-05-01
% Zhihong Zhang 2021.04.15	extent to multi-channel image, add border
%							selection

if(nargin < 6), border = 'on'; end

if ( N1 > 65535 ) || ( N2 > 65535 ) 
    error('The image size should be smaller than 65535 x 65535');
end


if ndims(M_patches)==3
	[n1, n2, num_patches] = size(M_patches);
	M_patches = reshape(M_patches,n1, n2, 1, num_patches);
end
[n1, n2, N3, num_patches] = size(M_patches);

if (delta1>n1) || (delta2>n2)
	warning('The step size is larger than the patch size, margin exists!')
end


% the coordinates of the top-left point in all the patches are computed and
% stored in (XstartI, YstartI). XstartI or YstartI is a vector of length
% #patches. 
Xstart = uint16(1 : delta1 : N1 - n1 + 1);
Ystart = uint16(1 : delta2 : N2 - n2 + 1);

% dealing with the border patches: if border='on' and the image can't be 
% divided exactly, keep the rest border patches by extend them from border
% to inside to form a complete patch.
if(strcmp(border,'on'))
    if(mod(N1-n1, delta1) ~= 0)
        Xstart = [Xstart  N1-n1+1]; 
    end
    if(mod(N2-n2, delta2) ~= 0)
        Ystart = [Ystart  N2-n2+1]; 
	end
else
	N1 = Xstart(end)+n1-1;
	N2 = Ystart(end)+n2-1;
	warning('BORDER off, the real image size is %dx%d\n',N1, N2);
end

[XstartI,YstartI] = ndgrid(Xstart,Ystart);
YstartI = YstartI(:);
XstartI = XstartI(:);

n1_minus1 = n1 - 1;
n2_minus1 = n2 - 1;

% real image size (border croped)
im = zeros(N1, N2, N3);
M_weight = zeros(N1, N2);

% use (one-layer) loop to extract the patches. This loop is inevitable in
% the reconstruction phase (patches2image) because we need to add the
% patches and accumulate the weight. 
for k = 1 : num_patches
    coor_x = XstartI(k):XstartI(k)+n1_minus1;
    coor_y = YstartI(k):YstartI(k)+n2_minus1;
    im(coor_x, coor_y,:) = im(coor_x, coor_y,:) + single(M_patches(:, :,:,k));
    M_weight(coor_x, coor_y) = M_weight(coor_x, coor_y) + 1;
end

im = im ./ M_weight;

