function z = cyl(N,R,P)
% CYL generates samples of a continuous, aperiodic, unity-height cylinder 
% at the points specified in P with radius R (percent of the length) on a background 
% rectangle with size N.
% 
%   Input:
%   --------
%   - N: 2-elements int vector, background rectangle size
%	- P: 2-elements int vector,centre point of the cylinder, default=[0,0]
%	- R: float scalar, radius of the cylinder
%   Output:
%   --------
%   - z: unity-height cylinder 

% input check
if numel(N)==1
	N = [N,N];		% background size
	z = zeros(N);	% background
elseif ~isvector(N)
	z = N;			% background
	N = size(z);	% background size
end

if nargin < 3
	P = round(N./2);
end

x0 = P(1);
y0 = P(2);
L1 = N(1);
L2 = N(2);

[x,y]=meshgrid(linspace(1, L1, L1), linspace(1, L2, L2)); 
x = x';
y = y';

r = sqrt((x-x0).*(x-x0)+(y-y0).*(y-y0)); % distance map

z(r<=R) = 1;
end