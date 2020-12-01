close all
[r,p,ar,ap] = matrix_corr(mask_bayer) % corr matrix and p-value
[corr_idx_matrix, corr_groups] = corr_group(p, -0.05);
% dmd_granularity = [6 6];
dmd_granularity = [3 3];

N = prod(dmd_granularity);
for k = 1:N
	corr_item  = cell2mat(corr_groups(k));
% 	[px, py] = ind2sub(dmd_blank_shape, [k corr_item])
	
	corr_matrix = zeros(dmd_granularity);
	corr_matrix(corr_item) = 1;
	corr_matrix(k) = 0.5;
% 	corr_matrix(px,py) = 1;
	
	figure
	imshow(corr_matrix)
	title(['correlation area(white) for pos ',num2str(k)]);
end