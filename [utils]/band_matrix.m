function band_mat = band_matrix(A, diag_ind, sz)
%BAND_MATRIX create band matrix according to the diagonal elements.
%
%	--------
%	Input
%	A: 2D matrix, vector or scalar. 
%		When 'A' is a 2D matrix, it represents a diagonal 
%	elements matrix, and each column of 'A' represents a diagonal of
%	elements. ! max(size(A)) should be >= min(SZ)
%		When 'A' is a vector, it means the band matrix has
%	equal elements in each diagonal, representing by each elements of
%	vector 'A'. 
%		When 'A' is a scalar, it means the band matrix has
%	equal elements in all non-zero diagonals, i.e. all = A, and the amount of 
%	non-zero diagonals is determined by DIAG_IND
% 
% 	diag_ind: the index of non-zero diagonals, the amount of DIAG_IND should be 
%	equal to the column number of 'A' e.g. -2:2
% 
%	sz: the size of band matrix
% 
%	--------
%	Output
%	mat: band matrix
% 
%   Info:
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-06-22
%   Last Modified:  Zhihong Zhang, 2020-06-22
%               
%   Copyright 2020 Zhihong Zhang

% input checking
if ismatrix(A) && isvector(diag_ind)
	if isscalar(A)
		A = ones(sz(1),numel(diag_ind))*A;
	elseif isrow(A)
		A = repmat(A, sz(1), 1);
	elseif iscolumn(A)
		A = repmat(A, 1, sz(1));
	end
else
	error('diagonal matrix - ''A'' error');
end

if ismatrix(A)
	if isscalar(sz)
		sz = sz*ones(1,2);
	end
else
	error('sz error');
end

if max(size(A)) < min(sz)
	error('max(size(A)) should be >= min(SZ)')
end

tmp_mat=spdiags(A, diag_ind, sz(1), sz(2));
band_mat = full(tmp_mat);

end

