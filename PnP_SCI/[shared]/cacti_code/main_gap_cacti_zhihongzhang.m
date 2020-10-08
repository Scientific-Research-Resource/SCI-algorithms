clear all 
close all
clc

%format short

% new algorithm for CACTI using TV denoising for each frame
% Xin Yuan; Bell Labs, Alcatel-Lucent
% xyuan@bell-labs.com
% initial date: 2015-7-02
% fname = 'traffic';
% load('./Yang_Liu_data/traffic48_cacti');

% fname = 'drop';
% load('./Yang_Liu_data/drop40_cacti');

% fname = 'kobe';
% load('./Yang_Liu_data/kobe32_cacti');


% fname = 'drop';
% load('D:\Zhihong_Zhang\PnP_Multiplex_Mask_20200828_zhihong\dataset\data\combine_binary_mask_256_10f\bm_256_10f\data_drop');


fname = 'kobe';

load(['E:\project\CACTI\SCI algorithm\[dataset]\#benchmark\data\combine_binary_mask_256_10f\bm_256_10f\data_' fname '.mat']);

addpath(genpath('./funs'));
row = 256;
col = 256;
Xtst = orig(1:row,1:col,1:1:end);  % original video

CodeFrame = 1;  % How many coded images, measurement
ColT = 10;  % How many frames are collasped into one measurement
[Row, Col, Frame] = size(orig);


% generate binary code
% Phi = rand(Row, Col,ColT);
 
% Phi(:,:,1) = binornd(1,0.5,[Row, Col]);
Phi = mask;

 % generate measurement
%  y = zeros(Row,Col,CodeFrame);
% for t = 1:CodeFrame
%    y(:,:,t) = sum(Xtst(:,:,(t-1)*ColT+(1:ColT)).*Phi,3);
% end

%% set parameters
para.row = row;
para.col = col;
para.iter = 200;
para.lambda = 0.2;  
para.TVweight = 0.05;

A = @(z) A_xy(z, Phi);
%At = @(z) At_xy(z, Phi,Phi_sum);
At = @(z) At_xy_nonorm(z, Phi);

 Phi_sum = sum(Phi.^2,3);
Phi_sum(Phi_sum==0)=1;
 para.lambda = 1;
 para.Phi_sum = Phi_sum;
 %para.iter =150;
 para.acc= 1;
 para.mu = 0.25;
 para.TV_iter =100;
 
 para.sigma   = [50 20 10 6]./255; % noise deviation (to be estimated and adapted)
  para.vrange   = 1; % range of the signal
  %para.maxiter = [ 40 40 60 ]*2;
  para.maxiter = [ 20 40 100 200];
  % para.maxiter = [10 10 10 10]; % first trial
 
TV_m_all = {'ATV_ClipA', 'ATV_ClipB','ATV_cham','ATV_FGP','ITV2D_cham','ITV2D_FGP','ITV3D_cham','ITV3D_FGP'};
  
addpath(genpath('E:\project\CACTI\SCI algorithm\PnP_SCI\matlab\packages\denoiser\'));
  
load(fullfile('models','FFDNet_gray.mat'));
para.net =  vl_simplenn_move(vl_simplenn_tidy(net), 'gpu') ;
  
  
for k=1:CodeFrame        
        fprintf('----- Reconstructing frame-block %d of %d\n',k,CodeFrame);
        para.ori_im = orig(:,:,(k-1)*ColT+(1:ColT))./255;
        y_use = meas(:,:,k)./255;
        %theta    =   DCT_CACTI_weight( y_use, para, A,At);
        
        %% Following algorithms are ATV
         for nm  =8 %1:length(TV_m_all)
             %para.TVweight = 0.05;
%              para.iter =150;
%                para.eta = 0.5;
%              para.TVm = TV_m_all{nm};
% %              if(strcmp(para.TVm,'ATV_ClipB'))
% %                  para.iter = 40;
% %              end
%              theta_admm{nm}(:,:,(k-1)*ColT+(1:ColT)) = TV_ADMM_CACTI_all( y_use, para,A,At); 
%          
             para.TVweight = 0.05;
            para.iter =150;
             para.TVm = TV_m_all{nm};
            if(strcmp(para.TVm,'ATV_ClipB'))
                para.iter = 40;
            end
            para.ori_im = single(orig(:,:,(k-1)*ColT+(1:ColT))./255);
            % theta_gaptv(:,:,(k-1)*ColT+(1:ColT)) = TV_GAP_CACTI_all( double(y_use), para,A,At);  % zzh
			
            %  theta_gap{nm}(:,:,(k-1)*ColT+(1:ColT)) = TV_GAP_CACTI_FFDnetfast( y_use, para,A,At); 
              
%               tic
%                theta_gapgpu{nm}(:,:,(k-1)*ColT+(1:ColT)) = TV_GAP_CACTI_FFDnetGPU( y_use, para,A,At); 
%                toc
               
               tic
               theta_gapgpu_joint(:,:,(k-1)*ColT+(1:ColT)) = TV_GAP_CACTI_FFDnetGPU_joint( y_use, para,A,At); 
               toc
               
%                tic
%                theta_gapgpu_jointsum(:,:,(k-1)*ColT+(1:ColT)) = TV_GAP_CACTI_FFDnetGPU_jointsum( y_use, para,A,At); 
%                toc
         
         
             
         end
        %%
  
  
%         img_gap(:,:,(k-1)*ColT+(1:ColT)) = theta;
%         img_gap_ATV_clipB(:,:,(k-1)*ColT+(1:ColT)) = thetaATV_clipB;
%         img_gap_ATV_cham(:,:,(k-1)*ColT+(1:ColT)) = thetaATV_cham;
%         img_gap_ATV_FGP(:,:,(k-1)*ColT+(1:ColT)) = thetaATV_FGP;
%         
%         img_gap_ITV2D_cham(:,:,(k-1)*ColT+(1:ColT)) = thetaITV2D_cham;
%         img_gap_ITV2D_FGP(:,:,(k-1)*ColT+(1:ColT)) = thetaITV2D_FGP;
%         
%         img_gap_ITV3D_cham(:,:,(k-1)*ColT+(1:ColT)) = thetaITV3D_cham;
%         img_gap_ITV3D_FGP(:,:,(k-1)*ColT+(1:ColT)) = thetaITV3D_FGP;
      
end

% figure; 
% for nm = 1:5
%     subplot(2,5,nm); imshow(theta_gapgpu{8}(:,:,nm)); title(psnr(theta_gapgpu{8}(:,:,nm), single(orig(:,:,nm))./255))
%     subplot(2,5,nm+5); imshow(theta_gapgpu{8}(:,:,nm+5)); title(psnr(theta_gapgpu{8}(:,:,nm+5), single(orig(:,:,nm+5))./255))
% end

% zzh
% figure; 
% for nm = 1:5
%     subplot(2,5,nm); imshow(theta_gaptv(:,:,nm)); title(psnr(theta_gaptv(:,:,nm), single(orig(:,:,nm))./255))
%     subplot(2,5,nm+5); imshow(theta_gaptv(:,:,nm+5)); title(psnr(theta_gaptv(:,:,nm+5), single(orig(:,:,nm+5))./255))
% end

figure; 
for nm = 1:5
    subplot(2,5,nm); imshow(theta_gapgpu_joint(:,:,nm)); title(psnr(theta_gapgpu_joint(:,:,nm), single(orig(:,:,nm))./255))
    subplot(2,5,nm+5); imshow(theta_gapgpu_joint(:,:,nm+5)); title(psnr(theta_gapgpu_joint(:,:,nm+5), single(orig(:,:,nm+5))./255))
end

% figure; 
% for nm = 1:5
%     subplot(2,5,nm); imshow(theta_gapgpu_jointsum(:,:,nm)); title(psnr(theta_gapgpu_jointsum(:,:,nm), single(orig(:,:,nm))./255))
%     subplot(2,5,nm+5); imshow(theta_gapgpu_jointsum(:,:,nm+5)); title(psnr(theta_gapgpu_jointsum(:,:,nm+5), single(orig(:,:,nm+5))./255))
% end


% nf =1;
% for nm=1:8
%     psnr_fista_all(nm,nf) = psnr(theta_fista{nm}, orig./255);
%     psnr_twist_all(nm,nf) = psnr(theta_twist{nm}, orig./255);
%     psnr_gap_all(nm,nf) = psnr(theta_gap{nm}, orig./255);
%     psnr_admm_all(nm,nf) = psnr(theta_admm{nm}, orig./255);
% end

% save([fname '_GAPTV1_FFDnet_20191021.mat']); 
% 
%  vffd_drop = theta_gapgpu{8};
% % % 
% save('drop_ffd.mat','vffd_drop');
    % show result
    %figure;
    
%     figure;
% for t=1:ColT
%  subplot(5, ColT, t); imshow(theta(:,:,t)); title(['GAP-TV: psnr: ' num2str(psnr(theta(:,:,t), Xtst(:,:,t)))]);
%  subplot(5, ColT, t+ColT); imshow(theta_tv_cham(:,:,t)); title(['GAP-TV-Cham: psnr: ' num2str(psnr(theta_tv_cham(:,:,t), Xtst(:,:,t)))]);
%  %subplot(2, ColT, t+ColT); imshow(img_3dtv_hard(:,:,t)); title(['3DTV: psnr: ' num2str(PSNR_3dtv_hard(t)) ]);
%  subplot(5, ColT, t+2*ColT); imshow(theta_tv_cham3d(:,:,t)); title(['3DTV-Cham: psnr: ' num2str(psnr(theta_tv_cham3d(:,:,t), Xtst(:,:,t))) ]);
%   subplot(5, ColT, t+3*ColT); imshow(theta_clip3d (:,:,t)); title(['3DTV: psnr: ' num2str(psnr(theta_clip3d(:,:,t), Xtst(:,:,t))) ]);
%   subplot(5, ColT, t+4*ColT); imshow(theta_tv_dtdh_clip(:,:,t)); title(['DtDh-Clip: psnr: ' num2str(psnr(theta_tv_dtdh_clip(:,:,t), Xtst(:,:,t))) ]);
% end
% 
% 
%  %% 
% clear para 
% para.row = row;
% para.col = col;
% para.T = ColT;
% para.iter = 200;
% para.Phi = Phi;
% para.At = At;
% 
% % para.L1_3d = 0.008;
% % para.L2_3d = 0.008;
% para.L1_3d = 0.1; %.008;
% para.L2_3d = 0.05; %.008;
% para.L1_2d = 0.2;
% para.L2_2d = 0.07;
%  
% for k=1:CodeFrame        
%         fprintf('----- Reconstructing frame-block %d of %d\n',k,CodeFrame);
%         para.ori_im = Xtst(:,:,(k-1)*ColT+(1:ColT));
%         y_use = y(:,:,k);
%         %theta    =   DCT_CACTI_weight( y_use, para, A,At);
%         theta2    =   ATV_3D_CACTI( y_use,para);
%         %theta3    =   ATV_3D_CACTI_Hard( y_use,para);
%   
%   
%         img_3dtv(:,:,(k-1)*ColT+(1:ColT)) = theta2;
%         %img_3dtv_hard(:,:,(k-1)*ColT+(1:ColT)) = theta3;
%       
% end
% %  
%  %save('traffic_3dtv.mat','-v7.3','img_3dtv','para','img_gap');
%  
% %  addpath(genpath('./SSTV_reference'))
% %  theta    =   STTV_GAP_CACTI( y_use,1, para, A,At);
% % 
% 
% for t=1:ColT*CodeFrame 
%       PSNR_gap(t) = psnr(Xtst(:, :,t),img_gap(:,:,t));
%       PSNR_3dtv(t) = psnr(Xtst(:, :,t),img_3dtv(:,:,t));
% end
% 
%  figure;
% for t=1:ColT
%  subplot(2, ColT, t); imshow(img_gap(:,:,t)); title(['GAP-TV: psnr: ' num2str(PSNR_gap(t)) ]);
%  %subplot(2, ColT, t+ColT); imshow(img_3dtv_hard(:,:,t)); title(['3DTV: psnr: ' num2str(PSNR_3dtv_hard(t)) ]);
%   subplot(2, ColT, t+ColT); imshow(img_3dtv(:,:,t)); title(['SSTV: psnr: ' num2str(PSNR_3dtv(t)) ]);
% end
