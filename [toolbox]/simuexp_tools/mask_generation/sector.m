function sect_mask = sector(width, height, cen_x, cen_y, CenAng_s, CenAng_e, r_s, r_e)
% SECTOR - generate sector shape binary mask
% Input
%	- width, height: background rectangle's size, scalar
%	- CenAng_s, CenAng_e: start and end angle, scalar, [-180, 180]
%	- r_s, r_e: start (inner) and end (outter) radius, scalar
% 
% Output
%  - sect_mask: sector mask
% 

    % Initialization
    sect_mask = zeros(height, width);
    % Offsets referring to sector center
    off_x = [1:width] - cen_x;
    off_y = -1*([1:height] - cen_y);
    % 
    for i = 1:width
        for j = 1:height
            r = sqrt(off_x(i)^2 + off_y(j)^2);
            alpha = ( atan2( off_y(j), off_x(i) ) )*180/pi;
            % In sector: Angle & Radius
            if alpha >= CenAng_s && alpha <= CenAng_e && r >= r_s && r <= r_e
                sect_mask(j, i) = 1;
            end
        end
    end
end