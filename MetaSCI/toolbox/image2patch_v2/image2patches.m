function M_patches = image2patches(im, n1, n2, delta1, delta2, border)
% Transfer an image to patches of size n1 x n2. The patches are sampled
% from the images with a translating distance delta1 x delta2. The result
% M_patches is organized as a 3d or 4d (multi-channel image input) stack, with M_patches(:,:,<:,>i), represent
% the i-th patch.
% 
% Input
% im:				original image
% [n1 n2]:			the patch's size
% [delta1 delta2]:	the step size
% border:			whether keep borders of the image when it cannot be
%					separated exactly, 'on' | 'off', default='on'
% 
% Output
% M_patches:		result patches, [n1, n2, <channel_num,> patch_num]
% 
% Xin Yuan, 2015.05.01
% Zhihong Zhang 2021.04.15	extent to multi-channel image, add border
%							selection

if(nargin < 6), border = 'on'; end

[N1, N2, N3] = size(im);

if ( N1 > 65535 ) || ( N2 > 65535 ) 
    error('The image size should be smaller than 65535 x 65535');
end

if (delta1>n1) || (delta2>n2)
	warning('The step size is larger than the patch size, no overlap!')
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
end

[XstartI,YstartI] = ndgrid(Xstart,Ystart);
YstartI = YstartI(:);
XstartI = XstartI(:);

n1_minus1 = n1 - 1;
n2_minus1 = n2 - 1;

% use (one-layer) loop to extract the patches. This loop is inevitable in
% the reconstruction phase (patches2image) because we need to add the
% patches and accumulate the weight. 
num_patches = length(XstartI);
M_patches = zeros(n1,n2, N3, num_patches, class(im));
for k = 1 : num_patches
    patch1 = im(XstartI(k):XstartI(k)+n1_minus1, YstartI(k):YstartI(k)+n2_minus1,:);
    M_patches(:,:,:,k) = patch1;
	
end

M_patches = squeeze(M_patches);
end
