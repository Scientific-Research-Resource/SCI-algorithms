function  y = gradf(z, v, Phi)
% Gradient function of data fidelity term (f)
% z: orig^
% v: meas
% Phi: mask

% 	Cr = size(Phi,3);
	v_est = sum(Phi.*z, 3); % estimated meas
	
   y = bsxfun(@times, v_est - v, Phi);
end