function mat = multiplex_matrix(sz, multiplex_num)
% MULTIPLEX_MATRIX: create multiplexing matrix with given size and multiplex_num
% (with as large rank as possible)
% 
%   Input:
%   --------
%   - sz: multiplexing matrix's size, 1x2 array, [multiplexed_result_num, element
%   num]
% 
%   - multiplex_num: number of elements used for multiplexing, 1xn array [1, multiplexed_result_num
%   num]. Or scalar, means the same multiplex_num for each multiplexing.
%   default = round(sz(2)/2);
% 
%   Note��
%   --------
%	
%	See also:
%	MULTIPLEX_MASK
% 
%   Info��
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-10-24
%   Last Modified:  Zhihong Zhang <z_zhi_hong@163.com>, 2020-10-24
%   
%   Copyright 2020 Zhihong Zhang

% input check
if nargin<2
	multiplex_num = round(sz(2)/2)
end

if isscalar(multiplex_num)
	multiplex_num = multiplex_num*ones(sz(1));
else
	error('Not Implemented');
end


% generate
init_mat = eye(sz(2)); % identity matrix with size of element size
mat = zeros(sz);
k = 0;
while( rank(mat) < min(sz) )
	for i = 1:sz(1)
		idx = randperm(sz(2), multiplex_num(i));
		mat(i,:) = sum(init_mat(idx,:), 1);
	end
	k = k+1;
	
	if(k>1e8)
		error("haven't find satisfied matrix after 1e8 iteration")
	end
end


end