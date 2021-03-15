%%
clc, clear, close all

% border = 'on'; skip_size = [64 64]; patch_size = [256 256];
% border = 'on'; skip_size = [64 40]; patch_size = [256 256];
% border = 'on'; skip_size = [40 40]; patch_size = [256 256];
% border = 'off'; skip_size = [100 100]; patch_size = [200 200];
border = 'off'; skip_size = [64 64]; patch_size = [256 256];

img = im2double(imread('einstein.bmp'));  % gray image
% img = im2double(imread('tower.jpg'));  % rgb image

img_size = size(img);

% im2patch
patches = im2patch(img, patch_size, skip_size, border);

% show patches
figure
if strcmp(border, 'on')
    sub_num = ceil((img_size(1:2)-patch_size)./skip_size)+1;
else
    sub_num = floor((img_size(1:2)-patch_size)./skip_size)+1;
end

channel_num = size(img,3);
patch_num = size(patches,ndims(patches));
for i  = 1:patch_num
	subplot(sub_num(1), sub_num(2), i)
    if channel_num==1
        imshow(patches(:,:,i))
    elseif channel_num==3
        imshow(patches(:,:,:,i))
    else
        disp('can not disp 4D data or data with more dimensions');
    end
end


%patch2img
img_recon = patch2im(patches, img_size, skip_size, border);

% compare
figure
subplot(2,1,1)
imshow(img);
subplot(2,1,2)
imshow(img_recon);


