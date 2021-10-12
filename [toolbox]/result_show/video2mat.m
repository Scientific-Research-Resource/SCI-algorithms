%% Convert video into mat file

% path
vid_name = 'car.mp4';
vid_dir = 'E:\project\SCI_captioning\code\tracking\CenterTrack\dataset\videos\all\tinghua\';
save_name = [vid_name(1:end-4) '.mat'];
save_dir = 'E:\project\SCI_captioning\code\SCI\PnP_SCI_python\dataset\orig\rgb\';

% param
frame_sz = [500 900];

% load video
v=VideoReader([vid_dir vid_name]);
frames = [];
frame_idx = 161:1:220;

for k = frame_idx
    frame = read(v,k);
	frame = imresize(frame,frame_sz);
    imshow(frame);
	title(['Frame #' num2str(k)])
    if isempty(frames)
        frames=frame;
    else
        frames(:,:,:,end+1) = frame;
	end
	drawnow
	
	if k>v.FrameRate*v.Duration
		disp("video endding")
		return
	end
end

% process
orig = frames;


% save
save([save_dir save_name], 'orig')
disp(['result saved to: ' save_dir save_name])