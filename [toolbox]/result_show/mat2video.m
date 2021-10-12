%% Convert video into mat file

% path
% mat_name = 'car.mat';
mat_name = 'gapfastdvdnet_car_kmeas0_5.mat';
% mat_dir = 'E:\project\SCI_captioning\code\SCI\PnP_SCI_python\dataset\orig\rgb\';
mat_dir = 'E:\project\SCI_captioning\code\SCI\PnP_SCI_python\results\savedmat\';
save_name = 'car_sci10.mp4';
save_dir = 'E:\project\SCI_captioning\code\tracking\CenterTrack\dataset\videos\pipeline_test\';

% param
vid_format = 'MPEG-4';
FrameRate = 10;

% load mat
mat_file = load([mat_dir mat_name]);
% mat_file = mat_file.orig;
mat_file = mat_file.vdenoise;

% process
mat_file = uint8(255*mat_file);

% write video
v=VideoWriter([save_dir save_name], vid_format);
v.FrameRate = FrameRate;

open(v)

for k = 1:size(mat_file,ndims(mat_file))
	writeVideo(v,mat_file(:,:,:,k))
end


close(v);
disp(['result saved to: ' save_dir save_name])