function [corr_idx_matrix, corr_group] = corr_group(corr_matrix, corr_level)
%CORR_GROUP get correlated groups from given correlation coefficient matrix (or p-value matrix)
%	
%   Input:
%   --------
%   - corr_matrix: correlation coefficient matrix (or p-value matrix), N*N
%   matrix
% 
%   - corr_level: double scalar, if corr_level>0, then it means value above "corr_level"
%   will be considered to be corelated (correlation coefficient matrix); if corr<0, 
%   the it means value below "corr_level" will be considered to be
%   corelated. (for p-value matrix)
% 
%   Output:
%   --------
%   - corr_group: correlation groups for every signal item (each row or each col, which represents
%   a signal item), 1*N cell
%	- corr_idx_matrix: correlation index matrix, 1-independent; 0-correlated
% 
%   Info:
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-04-27
%   Last Modified:  Zhihong Zhang, 2020-04-27
% 
%   Copyright (c) 2020 Zhihong Zhang.

corr_idx_matrix = corr_matrix;


if corr_level<0
	corr_idx_matrix(corr_matrix < -corr_level) = 0;  % correlated
	corr_idx_matrix(corr_matrix >= -corr_level) = 1; % independent
else
	corr_idx_matrix(corr_matrix > corr_level) = 0;
	corr_idx_matrix(corr_matrix <= corr_level) = 1;	
end

% diagonal elements are self-correlated for sure
N = size(corr_idx_matrix, 1); % corr_idx_matrix number of rows
corr_idx_matrix(eye(N, 'logical')) = 0;

figure,imshow(corr_idx_matrix)
title('corr-idx-matrix:1-independent; 0-correlated')


corr_group = cell(1,N); 

for k=1:N
	corr_group{k} = find(corr_idx_matrix(k,:)==0);
	corr_group{k} = [k, corr_group{k}];
end
end



