function [input_image, input_features, input_labels] = mapping_approach(file_path)

    T_jets = parquetread(file_path);
    
    T_jets = movevars(T_jets, 'is_signal_new', 'Before', 'E_0');
    T_jets = removevars(T_jets, 'x__index_level_0__');
    
    T_jets = sortrows(T_jets, 'is_signal_new');            % all background jets are above the top jets
    G = groupsummary(T_jets,'is_signal_new');
    
    Jet = T_jets_to_3d_tensor(T_jets);
    
    size(T_jets);
    
    function y = T_jets_to_3d_tensor(x)
        y = x{:, 2 : 4 * 35 + 1}';
        y = reshape(y, 4, 35, []);
        y = pagetranspose(y);
    end
    
    size(Jet);
    
    pT = sqrt(Jet(:, 2, :).^ 2 + Jet(:, 3, :).^ 2);         % transverse mom calculation
    eta = squeeze(asinh(Jet(:, 4, :) ./ pT));               % pseudorapidity tensor
    size(eta);
    phi = squeeze(atan2(Jet(:, 3, :), Jet(:, 2, :)));       % azimuthal tensor
    size(phi);
    
    pT = squeeze(pT);
    size(pT);
    
    eta = eta - eta(1, :);                                % centering each jet acoording to the first constituent, this makes the first constituent fall on the center of the image
    phi = phi - phi(1, :);
    
    Nb = 37;
    Xedges = linspace(-1.6, 1.6, Nb + 1);
    Yedges = linspace(-1.6, 1.6, Nb + 1);

    % this is the snippet for my mapping approach. For each jet, we map the coordinates where any constituent falls to that constituent index in the 'Jet' array.
    % This reduces the storage space required for creating the images in the later phase as we do NOT map all coordinates in the 37 * 37 region to a value, 
    % we map only the coordinates that actually have any constituent in them (such coordinaes are very less, around 15 in 1369 (37 ^ 2)) 
    
    pixel_maps = cell(1, size(Jet, 3));
    
    for n = 1: size(Jet, 3)
        row_idx = discretize(eta(:, n), Xedges);                             % finds the row index of the constituents
        col_idx = discretize(phi(:, n), Yedges);                             % finds the column index of the constituents
        pixel_map = containers.Map('KeyType', 'char', 'ValueType', 'any');   
    
        for i = 1: 35 
            r = row_idx(i);
            c = col_idx(i);
            if ~isnan(r) && ~isnan(c)
                key = sprintf('%d, %d', r, c);
                if isKey(pixel_map, key)                                     % check if the key is mapped to any other constituent, create a list of all constituents in that pixel
                    pixel_map(key) = [pixel_map(key), i];
                else                                                         % if not, map that coordinate to that constituent.
                    pixel_map(key) = i;
                end
            end
        end
        pixel_maps{n} = pixel_map;
    end
    
    num_jets = length(pixel_maps);
    
    % pixel wise channels
    particle_count_channel = zeros(Nb, Nb, num_jets);                    % stores number of particles per pixel
    pT_sum_channel = zeros(Nb, Nb, num_jets);                            % stores sum of transverse mom of all particles that fall into that coordinate, per coordinate/pixel
    pT_square_sum_channel = zeros(Nb, Nb, num_jets);                     % stores sum of transverse mom squared of all particles that fall into that coordinate, per coordinate/pixel
    energy_sum_channel = zeros(Nb, Nb, num_jets);                        % stores sum of energy of all particles that fall into that coordinate, per coordinate/pixel
    % momentum_magnitude_channel = zeros(Nb, Nb, num_jets);              % Discontinued the use of this channel
    energy_variance_channel = zeros(Nb, Nb, num_jets);                   % in pixels with more than one constituent, it finds the variance in energy and stores them
    pT_variance_channel = zeros(Nb, Nb, num_jets);                       % in pixels with more than one constituent, it finds the variance in pT and stores them
    energy_skewness_pixel_channel = zeros(Nb, Nb, num_jets);             % in pixels with more than one constituent, it finds the skewness in energy and stores them
    pT_skewness_pixel_channel = zeros(Nb, Nb, num_jets);                 % in pixels with more than one constituent, it finds the skewness in pT and stores them
    energy_kurtosis_pixel_channel = zeros(Nb, Nb, num_jets);             % in pixels with more than one constituent, it finds the kurtosis in energy and stores them
    pT_kurtosis_pixel_channel = zeros(Nb, Nb, num_jets);                 % in pixels with more than one constituent, it finds the kurtosis in pT and stores them
    % energy_pT_correlation_channel = zeros(Nb, Nb, num_jets);           % Discontinued the use of this channel
    avg_energy_channel = zeros(Nb, Nb, num_jets);                        % stores average of energy of all particles that fall into that coordinate, per coordinate/pixel
    angular_momentum_sum_channel = zeros(Nb, Nb, num_jets);              % stores the sum of angular momentum of 
    
    % image scale channels
    energy_skewness_global_channel = zeros(1, num_jets);                 % calculates how skewed the energy values are per jet(positive/negative skewed distribution of energy)
    pT_skewness_global_channel = zeros(1, num_jets);                     % calculates how skewed the pT values are per jet(positive/negative skewed distribution of energy)
    energy_kurtosis_global_channel = zeros(1, num_jets);                 % calculates how sharp or flat the energy values are in the middle per jet
    pT_kurtosis_global_channel = zeros(1, num_jets);                     % calculates how sharp or flat the pT values are in the middle per jet
    
    for n = 1: num_jets
        map = pixel_maps{n};
        keys = map.keys;
        e = squeeze(Jet(:, 1, n));
        energy_skewness_global_channel(n) = skewness(e);             
        pT_skewness_global_channel(n) = skewness(pT(:, n));
        energy_kurtosis_global_channel(n) = kurtosis(e);
        pT_kurtosis_global_channel(n) = kurtosis(pT(:, n));
    
        for k = 1: length(keys)
            key = keys{k};
            inds = map(key);
            parts = sscanf(key, '%d, %d');
            r = parts(1);
            c = parts(2);
    
            if r >= 1 && r <= Nb && c >= 1 && c <= Nb
                particle_count_channel(r, c, n) = numel(inds);
                pT_sum = 0;
                pT_square_sum = 0;
                E_sum = 0;
                % mom_sum = 0;
                angular_momentum_sum = 0;
                energy = zeros(1, 35);
                pt = zeros(1, 35);
                it = 0;
                for i = 1: length(inds)
                    ind = inds(i);
                    pT_sum = pT_sum + pT(ind, n);
                    pT_square_sum = pT_square_sum + pT(ind, n) ^ 2;
                    E_sum = E_sum + Jet(ind, 1, n);
                    % mom_sum = mom_sum + sqrt(Jet(ind, 2, n) ^ 2 + Jet(ind, 3, n) ^ 2 + Jet(ind, 4, n) ^ 2); 
                    it = it + 1;
                    energy(it) = Jet(ind, 1, n);
                    pt(it) = pT(ind, n);
    
                    r_dist = sqrt(eta(ind, n) ^ 2 + phi(ind, n) ^ 2);
                    mom_eq = sqrt(pT(ind, n) ^ 2 + Jet(ind, 4, n) ^ 2);
                    angular_momentum_sum = angular_momentum_sum + (r_dist * mom_eq);
                end
                pT_sum_channel(r, c, n) = pT_sum;
                pT_square_sum_channel(r, c, n) = pT_square_sum;
                energy_sum_channel(r, c, n) = E_sum;
                % momentum_magnitude_channel(r, c, n) = mom_sum;
                angular_momentum_sum_channel(r, c, n) = angular_momentum_sum;
                
                if n == 1
                    % disp(pt);
                    % disp(energy);
                    nnz(~(energy == 0 & pt == 0));
    
                    std(pt);
                    std(energy);
                    sum(energy == 0);
                    sum(pt == 0);
                end
    
                avg_energy_channel(r, c, n) = E_sum / length(inds);
    
                if length(inds) > 1          
                    energy_variance_channel(r, c, n) = var(energy);
                    pT_variance_channel(r, c, n) = var(pt);
                    energy_skewness_pixel_channel(r, c, n) = skewness(energy);
                    energy_kurtosis_pixel_channel(r, c, n) = kurtosis(energy);
                    pT_skewness_pixel_channel(r, c, n) = skewness(pt);
                    pT_kurtosis_pixel_channel(r, c, n) = kurtosis(pt);
                else
                    energy_skewness_pixel_channel(r, c, n) = 0;
                    energy_kurtosis_pixel_channel(r, c, n) = 0;
                    pT_skewness_pixel_channel(r, c, n) = 0;
                    pT_kurtosis_pixel_channel(r, c, n) = 0;
                end
            end
        end
    
    end
    
    
    particle_count_channel = zscore(particle_count_channel,0,3);             
    pT_sum_channel = zscore(pT_sum_channel,0,3);  
    pT_square_sum_channel = zscore(pT_square_sum_channel, 0, 3);
    energy_sum_channel = zscore(energy_sum_channel, 0, 3);
    energy_variance_channel = zscore(energy_variance_channel, 0, 3);
    pT_variance_channel = zscore(pT_variance_channel, 0, 3);
    energy_skewness_pixel_channel = zscore(energy_skewness_pixel_channel, 0, 3);
    pT_skewness_pixel_channel = zscore(pT_skewness_pixel_channel, 0, 3);
    energy_kurtosis_pixel_channel = zscore(energy_kurtosis_pixel_channel, 0, 3);
    pT_kurtosis_pixel_channel = zscore(pT_kurtosis_pixel_channel, 0, 3);
    avg_energy_channel = zscore(avg_energy_channel, 0, 3);
    angular_momentum_sum_channel = zscore(angular_momentum_sum_channel, 0, 3);
        
    for n= 1: num_jets
        particle_count_channel(:, :, n) = align_img(particle_count_channel(:, :, n), n, eta, phi, Xedges, Yedges);
        pT_sum_channel(:, :, n) = align_img(pT_sum_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        pT_square_sum_channel(:, :, n) = align_img(pT_square_sum_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        energy_sum_channel(:, :, n) = align_img(energy_sum_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        energy_variance_channel(:, :, n) = align_img(energy_variance_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        pT_variance_channel(:, :, n) = align_img(pT_variance_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        energy_skewness_pixel_channel(:, :, n) = align_img(energy_skewness_pixel_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        pT_skewness_pixel_channel(:, :, n) = align_img(pT_skewness_pixel_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        energy_kurtosis_pixel_channel(:, :, n) = align_img(energy_kurtosis_pixel_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        pT_kurtosis_pixel_channel(:, :, n) = align_img(pT_kurtosis_pixel_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        avg_energy_channel(:, :, n) = align_img(avg_energy_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        angular_momentum_sum_channel(:, :, n) = align_img(angular_momentum_sum_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
    
        fprintf('n: %f\n', n);
        ener_sum_top = sum(energy_sum_channel(1:18, :, n), 'all');
        ener_sum_bot = sum(energy_sum_channel(19:37, :, n), 'all');

        if ener_sum_bot < ener_sum_top
            flip_channels = {
                'particle_count_channel',
                'pT_sum_channel',
                'pT_square_sum_channel',
                'energy_sum_channel',
                'energy_variance_channel',
                'pT_variance_channel',
                'energy_skewness_pixel_channel',
                'pT_skewness_pixel_channel',
                'energy_kurtosis_pixel_channel',
                'pT_kurtosis_pixel_channel',
                'avg_energy_channel',
                'angular_momentum_sum_channel'
            };

            for k = 1:length(flip_channels)
                eval([flip_channels{k} ' = flip(' flip_channels{k} ', 1);']);
            end
        end

    end
    % visualize_averaged(particle_count_channel, G);  
        
    input_image = zeros(37, 37, 12, num_jets);
    
    input_image(:, :, 1, :) = particle_count_channel;
    input_image(:, :, 2, :) = pT_sum_channel;
    input_image(:, :, 3, :) = pT_square_sum_channel;
    input_image(:, :, 4, :) = energy_sum_channel;
    input_image(:, :, 5, :) = energy_variance_channel;
    input_image(:, :, 6, :) = pT_variance_channel;
    input_image(:, :, 7, :) = energy_skewness_pixel_channel;
    input_image(:, :, 8, :) = pT_skewness_pixel_channel;
    input_image(:, :, 9, :) = energy_kurtosis_pixel_channel;
    input_image(:, :, 10, :) = pT_kurtosis_pixel_channel;
    input_image(:, :, 11, :) = avg_energy_channel;
    input_image(:, :, 12, :) = angular_momentum_sum_channel;
    
    input_labels = T_jets.is_signal_new;
    
    input_features = zeros(1, 1, 4, num_jets);
    input_features(:, :, 1, :) = energy_skewness_global_channel;
    input_features(:, :, 2, :) = energy_kurtosis_global_channel;
    input_features(:, :, 3, :) = pT_skewness_global_channel;
    input_features(:, :, 4, :) = pT_kurtosis_global_channel;

    data_norm = input_features; 
    
    for feature = 1:4
        feature_values = squeeze(input_features(1,1,feature,:)); 
        mu = mean(feature_values);
        sigma = std(feature_values);
    
        normalized = (feature_values - mu) / sigma;
    
        data_norm(1,1,feature,:) = reshape(normalized, [1 1 1 num_jets]);
    end

    input_features = data_norm;
end







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

