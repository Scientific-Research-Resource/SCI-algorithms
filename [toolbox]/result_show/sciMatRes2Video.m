%% load DL and generate video

%% init
close, clear, clc

%% param
filename = {'Football','Hummingbird', 'Jockey','ReadySteadyGo','YachtRide'}; %filename
Cr = 10;		% [10 20]
sz = 256;		% [256, 512, 1024]
meas_nums = [4 4 4 4 4];
% meas_nums = [2 2 2 2 2];

framerate = 8;
save_dir = '../video/';

% load('./E2ECNN/E2ECNN_old_copy.mat');  % E2E-CNN results
% load('./GAP-net/gapnet_recon_ziyi.mat'); % GAP-net results




%% run
% load mask
load(['../mask/multiplex_shift_binary_mask_' num2str(sz) '_' num2str(Cr) 'f.mat']);  % meas (256x256x4) and orig (256x256x32)
 
for nf = 1:length(filename)
	fname = filename{nf};
	meas_num  = meas_nums(nf);
	total_frame = meas_num*Cr;

	% load data

	%ground truth
	load(['../orig/' num2str(sz) '/' fname '_' num2str(sz) '.mat']);  % meas (256x256x4) and orig (256x256x32)

	% meas
	meas = zeros(sz,sz,meas_num);
	for mm = 1:meas_num
	 meas(:,:,mm) = sum(single(orig(:,:,(mm-1)*Cr+1:mm*Cr)).*mask,3);
	end

	% load GAP-TV results
	data_gaptv=load(['../Cr' num2str(Cr) '/' num2str(sz) '/gaptv_' fname '_' num2str(sz) '_Cr' num2str(Cr) '.mat']);  % here we have rec_gaptv
	rec_gaptv = data_gaptv.vdenoise;
	psnr_gaptv = data_gaptv.psnr_denoise;
	clear vdenoise

	% load BIRNAT results
	data_birnat=load(['../Cr' num2str(Cr) '/' num2str(sz) '/birnat_' fname '_' num2str(sz) '_Cr' num2str(Cr) '.mat']);  % here we have rec_gaptv
	rec_birnat = data_birnat.vdenoise;
	psnr_birnat = data_birnat.psnr_denoise;
	clear vdenoise

	% load PnP-FFDnet results
	data_gapffdnet = load(['../Cr' num2str(Cr) '/' num2str(sz) '/gapffdnet_' fname '_' num2str(sz) '_Cr' num2str(Cr) '.mat']);  % here we have vgapffdnet
	rec_gapffdnet = data_gapffdnet.vdenoise; 
	psnr_gapffdnet = data_gapffdnet.psnr_denoise;
	clear vdenoise

	% load PnP-TV+FastDVDNet results
	data_tvfastdvdnet = load(['../Cr' num2str(Cr) '/' num2str(sz) '/gaptv+fastdvdnet_' fname '_' num2str(sz) '_Cr' num2str(Cr) '.mat']);  % here we have vgapffdnet
	rec_tvfastdvdnet = data_tvfastdvdnet.vdenoise; 
	psnr_tvfastdvdnet = data_tvfastdvdnet.psnr_denoise;
	clear vdenoise		 

	
	vname = [save_dir 'mp4/' num2str(sz) '_Cr' num2str(Cr) '/' fname '_' num2str(sz) '_Cr' num2str(Cr) '.mp4'];
	gifname = [save_dir 'gif/'  num2str(sz) '_Cr' num2str(Cr) '/' fname '_' num2str(sz) '_Cr' num2str(Cr) '.gif'];
	vobj = VideoWriter(vname,'MPEG-4');
	vobj.FrameRate =framerate;
	open(vobj);

	h = figure('Position',[50 50 2200 1200]);
	set(gcf,'color','white');
	set(0,'DefaultAxesFontSize',12);
	set(0,'DefaultTextFontSize',12);
	%h = figure
	ncount = 0;
	for nf = 1:total_frame
			ncount = ncount+1;
			nmea = ceil(nf/Cr);

			subplot(2,3,1); imshow(orig(:,:,nf)); title(['Ground Truth #' num2str(nf)]);
			subplot(2,3,2); imshow(meas(:,:,nmea)./max(meas(:))); title(['Measurement #' num2str(nmea)]);
			subplot(2,3,3); imshow(rec_gaptv(:,:,nf)); title(['GAP-TV #' num2str(nf) ' PSNR ' sprintf('%.2f',psnr_gaptv(nf)) 'dB']);

			subplot(2,3,4); imshow(rec_gapffdnet(:,:,nf)); title(['PnP-FFDNet #' num2str(nf) ' PSNR ' sprintf('%.2f',psnr_gapffdnet(nf)) 'dB']);
			subplot(2,3,5); imshow(rec_tvfastdvdnet(:,:,nf)); title(['PnP-TV-FastDVDNet #' num2str(nf) ' PSNR ' sprintf('%.2f',psnr_tvfastdvdnet(nf)) 'dB']);
			subplot(2,3,6); imshow(rec_birnat(:,:,nf)); title(['BIRNAT #' num2str(nf) ' PSNR ' sprintf('%.2f',psnr_birnat(nf)) 'dB']);

			figtitle(fname,'fontweight','bold')

			frame = getframe(h);
		writeVideo(vobj,frame);
		[vgif,map] = rgb2ind(frame.cdata,256,'nodither');
		if ncount==1
			imwrite(vgif,map,gifname,'DelayTime',1/framerate,'LoopCount',inf); % save as gif (first frame)
		else
			imwrite(vgif,map,gifname,'DelayTime',1/framerate,'WriteMode','append'); % save as gif
		end

	end
	close(vobj);

end




%% sub function
function [ fth ] = figtitle(titlestring,varargin)
% FIGTITLE creates a title centered at the  top of a figure. This may be used 
% to add a title to a figure with several subplots.
% 
% 
%% Syntax
% 
% figtitle('TitleString')
% figtitle('TitleString','TextProperty','TextValue')
% h = figtitle(...)
% 
%
%% Description 
% 
% figtitle('TitleString') centers a title at the top of a figure and sets
% the figure name to TitleString. 
% 
% figtitle('TitleString','TextProperty',TextValue) formats the title with
% property name value pairs (e.g., 'FontSize',20)
%
% h = figtitle(...) returns a handle of the newly-created title. 
%
%% EXAMPLE 1: 
% 
% x = 1:.01:7; 
% y = sin(x); 
% 
% figure; 
% subplot(2,2,1)
% plot(3*x,y)
% title('Exp. 1') 
% 
% subplot(2,2,2)
% plot(x,2*y+x)
% title('Exp. 2') 
% 
% subplot(2,2,3)
% plot(x,y)
% title('Exp. 3') 
% 
% subplot(2,2,4)
% plot(x,2*y)
% title('Exp. 4') 
% 
% figtitle('My Experimental Results','fontweight','bold');
% 
%% EXAMPLE 2: A prettier example using ntitle: 
% 
% x = 1:.01:7; 
% y = sin(x); 
% 
% figure; 
% subplot(2,2,1)
% plot(3*x,y)
% ntitle('experiment 1','fontsize',12)
% box off
% 
% subplot(2,2,2)
% plot(x,2*y+x)
% ntitle('experiment 2','fontsize',12)
% box off
% 
% subplot(2,2,3)
% plot(x,-y+5*x)
% ntitle('experiment 3','fontsize',12)
% box off
% 
% subplot(2,2,4)
% plot(x,2*y-3*x)
% ntitle('experiment 4','fontsize',12);
% box off
% 
% figtitle(' My Experimental Results')
% 
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * % 
% 
% In many cases a figure title may overlap a subplot title 
% To reduce the possibility of a figure title overlapping subplot
% titles, try pairing this function with the ntitle function, which 
% is available on the Mathworks File Exchange here: 
% http://www.mathworks.com/matlabcentral/fileexchange/42114-ntitle
%
% 
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * % 
% Written by Chad A. Greene of the University of Texas at Austin
% Institute for Geophysics, July 2013. 
% 
% Updated August 2014 to include support for invisible figures 
% and now also sets the figure name to the title string. 
%
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * % 
% 
% See also title, text, and ntitle. 


% Get the handle of the current axes and properties:
hca = gca; 
fontsize = get(hca,'fontsize'); 

% Create a new set of axes the size of the entire figure: 
h = axes('position',[0 0 1 1],'units','normalized');

axes('Units','normalized',...
    'Position',[0 0 1 1],...
    'Visible','off',...
    'XTick',[],...
    'YTick',[],...
    'Box','off');

% Make a title: 
fth = text(.5,1,titlestring,...
    'units','normalized',...
    'horizontalalignment','center',...
    'verticalalignment','top',...
    'fontsize',fontsize+2); 

% Set optional inputs: 
if nargin>1
    set(fth,varargin{:});
end

% Now go back to from where we came: 
delete(h)

set(gcf,'CurrentAxes',hca,'name',titlestring); 

% Return the title handle only if it is desired: 
if nargout==0
    clear fth; 
end

end


