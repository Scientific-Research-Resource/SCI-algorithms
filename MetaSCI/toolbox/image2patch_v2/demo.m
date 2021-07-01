I = im2double(imread('einstein.bmp'));
I_sz = size(I);
M_patches = image2patches(I, 200, 200, 100, 100);
im = patches2image(M_patches, I_sz(1), I_sz(2),  100, 100);
figure,imshow(im,[])

M_patches = image2patches(I, 100, 100, 100, 100,'off');
im = patches2image(M_patches, I_sz(1), I_sz(2),  100, 100,'off');
figure,imshow(im,[])

I=imread('tower.jpg');
I_sz = size(I);
M_patches = image2patches(I, 200, 200, 40, 50);
im = patches2image(M_patches,I_sz(1), I_sz(2),  40, 50);
figure,imshow(uint8(im),[])

M_patches = image2patches(I, 200, 200, 40, 50,'off');
im = patches2image(M_patches,I_sz(1), I_sz(2),  40, 50,'off');
figure,imshow(uint8(im),[])