# image2patch
Convert image to patches or  recover image for patches with given size and stride.

> Author: Zhihong Zhang
>
> Email: z_zhi_hong@163.com

## Directory & Usage

- demo_patching.m: a demo for the patching and restorage functions
- im2patch: convert image to patches with given size and stride, and the border can be chose to keep or discard, parameters：
  - src_img: original image, gray or multi-channel
  - patch_size: size of small patches, 2D int
  - skip_size: the stride of patches, 2D int
  - border: whether keep borders of the image when it cannot be separated exactly, string array, value={'on', 'off'}
  - patches: small patches, [h, w, c] for gray image; [h, w, c, p] for multi-channel
- patch2im: recover image for patches with given image size, stride, and border information, parameters：
  - patches: small patches, [h, w, c] for gray image; [h, w, c, p] for multi-channel
  - img_size: the size of original image, 2D int
  - skip_size: the stride of patches, 2D int
  - border: whether keep borders of the image when it cannot be separated exactly, string array, value={'on', 'off'}
  -  img_recon: reconstructed image, gray or multi-channel

## Reference

- http://hvrl.ics.keio.ac.jp/charmie/file/demo_patching.zip