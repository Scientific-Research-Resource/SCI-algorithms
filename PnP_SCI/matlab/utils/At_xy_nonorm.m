function  y = At_xy_nonorm(z, Phi)
%z = z./Phi_sum;   
y = bsxfun(@times, z, Phi); 

% z: 'meas';
% y = {z*Phi(:,:1), z*Phi(:,:2),...}
end