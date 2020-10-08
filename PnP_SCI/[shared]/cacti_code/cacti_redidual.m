% residual learning for CACTI
clear all
close all
clc

data_file = './Yang_Liu_data/traffic48_cacti.mat';
load(data_file);
T = 8;

y_use = meas(:,:,1);
truth = orig(:,:,1:T);

mask_sum = sum(mask.^2,3);
mask_sum_1 = mask_sum;
mask_sum_1(mask_sum_1==0) = 1;
PhiPhiT = mask_sum;
PhiPhiT(PhiPhiT==0) = 1;

x_ave = y_use./mask_sum_1;
x_ave_TV =   tvdenoise_cham_ITV2D(x_ave,  0.1,5);  
 figure; imshow(x_ave_TV./max(x_ave_TV(:)))
% denosing using medfilt2
x_ave_med = medfilt2(x_ave, [5, 5]);
figure; imshowpair(x_ave,x_ave_med,'montage');


lambda        = 10^-9; 
tol           = 1e-8;
maxiter       = 4000;
dt            = 0.01;
% inpainting

addpath('./Inpainting')
inpain_mask =double( (x_ave<1e-5));
tic
x_ave_in = inpainting_amle(x_ave,inpain_mask,lambda,tol,maxiter,dt);
toc
figure; imshowpair(x_ave,x_ave_in,'montage');



x_ini_PhiT = bsxfun(@times, y_use, mask);

x_ini_PhiTPhiPhiTinv = bsxfun(@times, y_use./PhiPhiT, mask);


figure; 
for t=1:T
    temp = medfilt2(squeeze(x_ini_PhiTPhiPhiTinv(:,:,t)), [3, 3]);
    x_ini_PhiTPhiPhiTinv_med(:,:,t) = temp;
    %figure; imshowpair(x_ave,x_ave_med,'montage')
end


figure; 
for t = 1:T
    subplot(4,T,t); imshow(truth(:,:,t)./max(truth(:))); title(['Truth: ' num2str(t)]);
    subplot(4,T,t+T); imshow(x_ini_PhiT(:,:,t)./max(x_ini_PhiT(:))); title(['PhiTy: ' num2str(t)]);
    subplot(4,T,t+2*T); imshow(x_ini_PhiTPhiPhiTinv(:,:,t)./max(x_ini_PhiTPhiPhiTinv(:))); title(['PhiTPhiPhiTinvy: ' num2str(t)]);
     subplot(4,T,t+3*T); imshow(x_ini_PhiTPhiPhiTinv_med(:,:,t)./max(x_ini_PhiTPhiPhiTinv_med(:))); title(['PhiTPhiPhiTinvy_med: ' num2str(t)]);
end

figure;
for t=1:T
    subplot(3,T,t); imshow(truth(:,:,t)./max(truth(:))); title(['Truth: ' num2str(t)]);
    subplot(3,T,t+T); imshow(x_ave_med./max(x_ave_med(:)));
    subplot(3,T,t+2*T); imagesc(truth(:,:,t)-x_ave_med);
end


addpath(genpath('./funs'));

row = 256;
col = 256;
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


A = @(z) A_xy(z, Phi);
%At = @(z) At_xy(z, Phi,Phi_sum);
At = @(z) At_xy_nonorm(z, Phi);

 Phi_sum = sum(Phi.^2,3);
Phi_sum(Phi_sum==0)=1;
 para.lambda = 1;
 para.Phi_sum = Phi_sum;
 para.iter =150;
 para.acc= 1;
 para.mu = 0.25;
 para.ori_im = truth./255;
 para.x_ave = x_ave./255;
 para.TVweight = 0.07;
 theta    =   TV_GAP_CACTI( y_use./255,1, para, A,At);  % GAP-ATV clipA
 
 theta2 = theta.*255;
 theta2_clip = max(min(theta2,255),0);
 
  theta1    =   TV_res_GAP_CACTI( y_use./255,1, para, A,At);  % GAP-ATV clipA
  
  figure;
for t=1:T
    subplot(3,T,t); imshow(truth(:,:,t)./max(truth(:))); title(['Truth: ' num2str(t)]);
    subplot(3,T,t+T); imshow(theta(:,:,t)./max(theta(:)));
    subplot(3,T,t+2*T); imshow(theta1(:,:,t)./max(theta1(:)));
end


res = theta2_clip-truth;
figure; for t=1:T subplot(1,T,t); imagesc(res(:,:,t)); end
figure; for t=1:T subplot(1,T,t); tem = squeeze(res(:,:,t)); histogram(tem(:)); end