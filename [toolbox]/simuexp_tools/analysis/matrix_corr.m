function [corr_mat, p_mat, aver_corr, aver_p] = matrix_corr(X, varargin)
%MATRIX_CORR calculate the correlation coefficient of given matrix
%	convert the matrixs to column vector and calculate their correlation
%	coefficient matrix and average correlation coefficient
% 
%	Input:
%	-------
%	X:	input matrix, 2D or 3D numerical matrix (3D means matrix is stacked along 
%	3rd axis, Y is omitted in these case)
%	varargin: more input matrix, 2D numerical matrix, take effect when X is
%	also a 2D numerical matrix
%	
%	Output:
%	corr_mat:	correlation coefficient matrix of columns (straightened
%	matrix)
%	p_mat:	    p value matrix of columns
%	aver_corr:	average correlation coefficient of different columns
%	aver_p:	    average p-value of different columns
% 
%   Note:
%   --------
%   This funcion can be substituted with built-in function CORRCOEF
% 
% 
%   Info:
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-04-06
%   Last Modified:  Zhihong Zhang, 2020-04-06
% 
%   Copyright (c) 2020 Zhihong Zhang.


Y = cell2mat(varargin);

if nargin == 1 && isnumeric(X) && (ndims(X)==3)
	matrix_stack = X;
	
elseif nargin > 1 && ismatrix(Y) && (isnumeric(Y)||islogical(Y))
	
	Y = reshape(Y, [size(varargin{1}), numel(varargin)]);
	matrix_stack = cat(3,X,Y);
	
else
	error("multi inputs--2D matrixs or one input--single 3D matrix is valid inputs")
end

matrix_stack = double(matrix_stack); % int is not supported by corr
 
matrix_stack_sz =size(matrix_stack);
X2col = reshape(matrix_stack, [prod(matrix_stack_sz(1:2)) matrix_stack_sz(3)]);
[corr_mat, p_mat] = corrcoef(X2col);

corr_mat_triu = triu(corr_mat, 1);
aver_corr = sum(abs(corr_mat_triu), 'all')/nchoosek(matrix_stack_sz(3),2);

p_mat_triu = triu(p_mat, 1);
aver_p = sum(abs(p_mat_triu), 'all')/nchoosek(matrix_stack_sz(3),2);
end
