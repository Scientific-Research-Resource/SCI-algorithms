function  [im_denoised]     =  FFD_Net_DenoiserGPU(input, imageNoiseSigma,net)



inputNoiseSigma   =   imageNoiseSigma;

 input = gpuArray(input);
format compact;
global sigmas;



    
sigmas = inputNoiseSigma; 
    
 res    = vl_simplenn(net, input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default % use this if you did  not install matconvnet; very slow
    
     output= res(end).x;
     im_denoised= gather(output);



end

