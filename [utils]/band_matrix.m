function band_mat = band_matrix(A, diag_ind, size)
%BAND_MATRIX create band matrix according to the diagonal elements.
%
%	--------
%	Input
%	A: 2D matrix or vector. When 'A' is a 2D matrix, it represents a diagonal 
%	elements matrix, and each column of 'A' represents a diagonal of
%	elements. The row number of ¡®A' should equal to the row/col num of
%	given size. When 'A' is a vector ( or a scalar), it means the band matrix has
%	equal elements in each diagonal, representing by each elements of
%	vector 'A'
% 
% 	diag_ind: the index of non-zero diagonals, the amount of 'diag_ind' should be 
%	equal to the column number of 'A' e.g. -2:2
% 
%	size: the size of band matrix
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
	if isrow(A)
		A = repmat(A, size(1), 1);
	elseif iscolumn(A)
		A = repmat(A, 1, size(2));
	end
else
	error('diagonal matrix - ''A'' error');
end

tmp_mat=spdiags(A, diag_ind, size(1), size(2));
band_mat = full(tmp_mat);

end

