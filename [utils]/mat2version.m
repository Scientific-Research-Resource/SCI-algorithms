function mat2version(src_path, dst_path, version)
%MAT2VERSION convert .mat file to assigned version.
% 
%	Input
%	--------
%	src_path: source path, directory path or file path, str
%	dst_path: destination path, directory path or file path, str
%	version: saving version, str, e.g. '-v7.3', '-v7'
% 
%	Output
%	--------
%	directly saving the converted data to the 'dst_path'
% 
%   Info£º
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-06-05
%   Last Modified:  Zhihong Zhang <z_zhi_hong@163.com>, 2020-06-05
%   
%   Copyright 2020 Zhihong Zhang

src_file_flag = isfile(src_path);
dst_path = char(dst_path);
if strcmp(dst_path(end-3:end), '.mat')
	dst_file_flag = 1;
else
	dst_file_flag = 0;
end

if src_file_flag == 0 && dst_file_flag == 1
	error('dst_path should be a folder')
end

if src_file_flag
	% single file
	files = {src_path};
else
	% dir
	file_list = dir([src_path, '*.mat']);
	files = {file_list.name};
end

% convert
for file = files
	file_path = char(file);
	data = load(file_path);
	if dst_file_flag
		save(dst_path,'-struct', 'data', version)
	else
		if ~isfolder(dst_path)
			mkdir(dst_path)
		end
		[~,src_name, ~]= fileparts(file_path);
		save(fullfile(dst_path, src_name), '-struct', 'data', version)
	end
end