% calculate psnr & ssim for birnat

% init env
clear, clc

% param
filenames = {'Football','Hummingbird', 'Jockey','ReadySteadyGo','YachtRide'}; %filename
file_num = 5;
Crs = [10 20];
sizes = [256];


for Cr = Crs
	for sz = sizes
		for n = 1:file_num
			% load data and orig
			fname = filenames{n};
			datapath = ['../Cr' num2str(Cr) '/' num2str(sz) '/birnat_' fname '_' num2str(sz) '_Cr' num2str(Cr) '.mat'];
			origpath = ['../orig/' num2str(sz) '/' fname '_' num2str(sz) '.mat'];
			load(datapath)
			load(origpath)

			meas_num = size(pic,1);
			total_frame = meas_num*Cr;
			
			% cvt data format
			vdenoise = zeros(sz,sz, total_frame);
			for nm = 1:meas_num
				vdenoise(:,:,(nm-1)*Cr+(1:Cr)) = shiftdim(squeeze(pic(nm,:,:,:)),1);
			end
			
			% psnr ssim
			psnr_denoise = zeros(1, total_frame);
			ssim_denoise = zeros(1, total_frame);
			for k = 1:total_frame
				psnr_denoise(k) = psnr(vdenoise(:,:,k), double(orig(:,:,k))/255);
				ssim_denoise(k) = ssim(vdenoise(:,:,k), double(orig(:,:,k))/255);
			end
			psnr_mean = mean(psnr_denoise);
			ssim_mean = mean(ssim_denoise);
			savepath = ['./' 'birnat_' fname '_' num2str(sz) '_Cr' num2str(Cr) '.mat'];
			save(savepath,'vdenoise','psnr_denoise', 'ssim_denoise', 'psnr_mean', 'ssim_mean')
		end
	end
end
