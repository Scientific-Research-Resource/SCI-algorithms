%% data postprocess for image stitching
% TBD: generalization and refine

clc, clear;

%% params
root = '/data/zzh/project/SCI_cap/recon/20211122_lr256_trainmeasnoise_0.050000_realdata1122';
save_dir = [root '/full_real'];

w_num = 13;
h_num = 10;
y_max = 2110;
x_max = 2728;
scale = 1;
L = 256;
margin = 50;
L_m = L - margin*scale;
epoch = 0106;
iter = 4000;
scenes = dir(fullfile(root,'patch001/real/',num2str(epoch,'%04d'),['/iter_',num2str(iter)]));

%% processing
for scene_idx = 3:length(scenes)
    scene = scenes(scene_idx).name;
    meas_dirs = dir(fullfile(root,'patch001/real/',num2str(epoch,'%04d'),['/iter_',num2str(iter)],scene));
    for meas_dir_idx = 3:length(meas_dirs)
        meas_idx = meas_dirs(meas_dir_idx).name;
%         if mod(str2num(meas_idx),2) == 1
%             continue
%         end
        recon = zeros(y_max*scale,x_max*scale,8);
        
        mkdir(fullfile(save_dir, scene, meas_idx));
        for y = 1:h_num
            for x = 1:w_num
                patch_idx = (y - 1) * w_num + x;
%                 if patch_idx >= 108
%                     break
%                 end
                disp(patch_idx)
                recon_path = fullfile(root,sprintf('patch%s',num2str(patch_idx,'%03d')),'/real/',num2str(epoch,'%04d'),['/iter_',num2str(iter)],scene,meas_idx,sprintf('meas%s.mat',num2str(patch_idx,'%03d')));
                if exist(recon_path)
                    load(recon_path)
                else
                    warning(sprintf('No recon patch%03d, use zeros',patch_idx))
                    out_birnat = zeros(8,L,L);
                end
%                 figure,imshow(squeeze(out_birnat(1,:,:)),[]);
                
                out_ave = permute(out_birnat,[2,3,1]);
                if y ~= 1
                    out_ave(1:margin*scale,:,:) = (out_ave(1:margin*scale,:,:) + ...
                        recon((y-1)*L_m+1:(y-1)*L_m+margin*scale,(x-1)*L_m+1:(x-1)*L_m+L,:)) / 2;
                end
                if x ~= 1
                    out_ave(:,1:margin*scale,:) = (out_ave(:,1:margin*scale,:) + ...
                        recon((y-1)*L_m+1:(y-1)*L_m+L,(x-1)*L_m+1:(x-1)*L_m+margin*scale,:)) / 2;
                end
                recon((y-1)*L_m+1:(y-1)*L_m+L, (x-1)*L_m+1:(x-1)*L_m+L,:) = out_ave;
            end
        end

        enh = 1;
%         figure(2)
        y = 2;
        x = 4;
        
        
        save_path = fullfile(save_dir, scene, meas_idx, '1116.mat');
        save(save_path,'recon')        
        for i = 1:8
%             imshow(recon(:,:,i)*enh)
            
            save_path = fullfile(save_dir, scene, meas_idx, sprintf('1116_%d.png',i));
            imwrite(recon(:,:,i),save_path)
        %     save_crop_path = fullfile(save_dir, sprintf('1116_crop_%d.png',i));
        %     imwrite(recon(L*(y-1)+1:L*y,L*(x-1)+1:L*x,i),save_crop_path)
            title(i)
%             pause(0.2)
        end

        recon_2d = zeros(y_max*scale*2, x_max*scale*4);
%         figure(3)
        for y = 1:2
            for x = 1:4
                recon_2d((y-1)*y_max*scale+1:y*y_max*scale, (x-1)*x_max*scale+1:x*x_max*scale) = recon(:,:,(y-1)*4+x);
            end
        end
%         imshow(recon_2d)
        save_path = fullfile(save_dir, sprintf('%s_%s.png',scene,meas_idx));
        imwrite(single(recon_2d),save_path)
    end
end