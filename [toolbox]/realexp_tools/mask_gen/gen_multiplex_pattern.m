% setting
blk_sz = [6,8];
sz = [1536 2048];
Cr = 30;
save_name = 'mask';
save_dir = './mask2_Cr10_3#4/';

% gen
mask = zeros([sz Cr]);
[~, blk_pattern]=multiplex_matrix([Cr,prod(blk_sz)], prod(blk_sz)/2, blk_sz);
for k=1:Cr
   blk_mask = blk_pattern(:,:,k);
   mask_k = kron(blk_mask, ones(floor(sz./blk_sz)));
   mask(:,:,k) = imresize(mask_k, sz,'nearest');
end

% mkdir
if ~exist(save_dir, 'dir')
    mkdir(save_dir)
end

for i = 1:Cr
    save_path = sprintf([save_dir save_name '_%02d.png'], i);
    imwrite(uint8(255*mask(:,:,i)), save_path)
end
save_mat_path = sprintf([save_dir save_name '_Cr%d.mat'], Cr);
save(save_mat_path, 'mask') 