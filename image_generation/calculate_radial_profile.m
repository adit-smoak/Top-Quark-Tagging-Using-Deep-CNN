% an array of size 19 is created. we start from pixel 19, 19 -> store its energy in array index 1. then we move to the next 8 radially neighbouring pixels, 
% calculate the sum of energy /pT -> store it in array index 2. then next radially outward 16 pixels, calculate the sum of energy/pT -> store it in array index 3 
% and so on. This is done according to Chebyshev distance from the center. This creates a radial energy/pT distribution which is used for calculating 
% the global parameters.

function radial_array = calculate_radial_profile(image)
    radial_array = zeros(1, 19);
    center_row = 19;
    center_col = 19;
    
    radial_array(1) = image(center_row, center_col);
    
    for ring = 1:18
        ring_sum = 0;
        
        for row = 1:37
            for col = 1:37
                distance = max(abs(row - center_row), abs(col - center_col));
                if distance == ring
                    ring_sum = ring_sum + image(row, col);
                end
            end
        end
        
        radial_array(ring + 1) = ring_sum;
    end
end
