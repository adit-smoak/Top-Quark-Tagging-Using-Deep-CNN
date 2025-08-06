function [input_image, input_features, input_labels] = mapping_approach(file_path)

    T_jets = parquetread(file_path);
    
    T_jets = movevars(T_jets, 'is_signal_new', 'Before', 'E_0');
    T_jets = removevars(T_jets, 'x__index_level_0__');
    
    T_jets = sortrows(T_jets, 'is_signal_new');              % all background jets are above the top jets
    % G = groupsummary(T_jets,'is_signal_new');
    
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

    % This is the snippet for my mapping approach. For each jet, we map the coordinates where any constituent falls, to that constituent index in the 'Jet' array.
    % This reduces the storage space required for creating the images in the later phase, as we do NOT map all coordinates in the 37 * 37 region to a value, 
    % We map only the coordinates that actually have any constituent in them (such coordinates are very less, around 15 in 1369 (37 ^ 2)) 
    
    pixel_maps = cell(1, size(Jet, 3));
    
    % this 'for' loop creates 'num_jets' amount of pixel maps, where each map contains all the indices with 1 or more particles as keys and
    % the indices of constituents in an array as the value of that key.
    
    
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
    pz_sum_channel = zeros(Nb, Nb, num_jets);                            % stores sum of mom in z direction of all particles that fall into that coordinate, per coordinate/pixel
    energy_sum_channel = zeros(Nb, Nb, num_jets);                        % stores sum of energy of all particles that fall into that coordinate, per coordinate/pixel
    energy_variance_channel = zeros(Nb, Nb, num_jets);                   % in pixels with more than one constituent, it finds the variance in energy and stores them
    pT_variance_channel = zeros(Nb, Nb, num_jets);                       % in pixels with more than one constituent, it finds the variance in pT and stores them
    energy_skewness_pixel_channel = zeros(Nb, Nb, num_jets);             % in pixels with more than one constituent, it finds the skewness in energy and stores them
    pT_skewness_pixel_channel = zeros(Nb, Nb, num_jets);                 % in pixels with more than one constituent, it finds the skewness in pT and stores them
    energy_kurtosis_pixel_channel = zeros(Nb, Nb, num_jets);             % in pixels with more than one constituent, it finds the kurtosis in energy and stores them
    pT_kurtosis_pixel_channel = zeros(Nb, Nb, num_jets);                 % in pixels with more than one constituent, it finds the kurtosis in pT and stores them
    avg_energy_channel = zeros(Nb, Nb, num_jets);                        % stores average of energy of all particles that fall into that coordinate, per coordinate/pixel
    angular_momentum_sum_channel = zeros(Nb, Nb, num_jets);              % stores the sum of angular momentum of 

    % this 'for' loop calculates the values for each channel, like the sum of energy of all constituents in a pixel for energy_sum_channel, sum of pT of all
    % constituents in a pixel for pT_sum_channel etc.
    
    for n = 1: num_jets
        map = pixel_maps{n};
        keys = map.keys;
    
        for k = 1: length(keys)
            key = keys{k};
            inds = map(key);
            parts = sscanf(key, '%d, %d');
            r = parts(1);
            c = parts(2);
    
            if r >= 1 && r <= Nb && c >= 1 && c <= Nb
                particle_count_channel(r, c, n) = numel(inds);
                pT_sum = 0;
                pZ = 0;
                E_sum = 0;
                angular_momentum_sum = 0;
                energy = zeros(1, 35);
                pt = zeros(1, 35);
                it = 0;
                for i = 1: length(inds)
                    ind = inds(i);
                    pT_sum = pT_sum + pT(ind, n);
                    pZ = pZ + abs(Jet(ind, 4, n));
                    E_sum = E_sum + Jet(ind, 1, n);
                    it = it + 1;
                    energy(it) = Jet(ind, 1, n);
                    pt(it) = pT(ind, n);
    
                    r_dist = sqrt(eta(ind, n) ^ 2 + phi(ind, n) ^ 2);
                    mom_eq = sqrt(pT(ind, n) ^ 2 + Jet(ind, 4, n) ^ 2);
                    angular_momentum_sum = angular_momentum_sum + (r_dist * mom_eq);
                end
                pT_sum_channel(r, c, n) = pT_sum;
                pZ_sum_channel(r, c, n) = pZ;
                energy_sum_channel(r, c, n) = E_sum;
                angular_momentum_sum_channel(r, c, n) = angular_momentum_sum;
    
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

    % This is the preprocessing step where we check the second particle's location and rotate the image so that the second particle is in the angle range [-45°, 45°]
    % then the image if flipped vertically if the sum of energy of the top part (row 1 to 18) is more than the sum of the bottom part (row 20 to 37). 
    % The reason is described in the document in the literature folder.

    for n= 1: num_jets
        particle_count_channel(:, :, n) = align_img(particle_count_channel(:, :, n), n, eta, phi, Xedges, Yedges);
        pT_sum_channel(:, :, n) = align_img(pT_sum_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
        pZ_sum_channel(:, :, n) = align_img(pZ_sum_channel(:, :, n), n, eta, phi, Xedges, Yedges); 
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
                'pZ_sum_channel',
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

    % Perform z-score normalization across the 3rd dimension
    
    particle_count_channel = zscore(particle_count_channel,0,3);             
    pT_sum_channel = zscore(pT_sum_channel,0,3);  
    pZ_sum_channel = zscore(pZ_sum_channel, 0, 3);
    energy_sum_channel = zscore(energy_sum_channel, 0, 3);
    energy_variance_channel = zscore(energy_variance_channel, 0, 3);
    pT_variance_channel = zscore(pT_variance_channel, 0, 3);
    energy_skewness_pixel_channel = zscore(energy_skewness_pixel_channel, 0, 3);
    pT_skewness_pixel_channel = zscore(pT_skewness_pixel_channel, 0, 3);
    energy_kurtosis_pixel_channel = zscore(energy_kurtosis_pixel_channel, 0, 3);
    pT_kurtosis_pixel_channel = zscore(pT_kurtosis_pixel_channel, 0, 3);
    avg_energy_channel = zscore(avg_energy_channel, 0, 3);
    angular_momentum_sum_channel = zscore(angular_momentum_sum_channel, 0, 3);

    input_image = zeros(37, 37, 12, num_jets);
    
    input_image(:, :, 1, :) = particle_count_channel;
    input_image(:, :, 2, :) = pT_sum_channel;
    input_image(:, :, 3, :) = pZ_sum_channel;
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

    % global image features
    input_features = zeros(1, 1, 4, num_jets);

    for n = 1: num_jets
        energy_radial = calculate_radial_profile(energy_sum_channel(:, :, n));    % function to calculate the radial energy/pT distribution in the image
        input_features(1, 1, 1, n) = skewness(energy_radial);
        input_features(1, 1, 2, n) = kurtosis(energy_radial);
    
        pT_radial = calculate_radial_profile(pT_sum_channel(:, :, n));
        input_features(1, 1, 3, n) = skewness(pT_radial);
        input_features(1, 1, 4, n) = kurtosis(pT_radial);
    end

    data_norm = input_features; 

    % perform normalization of the features
    for feature = 1:4
        feature_values = squeeze(input_features(1,1,feature,:)); 
        mu = mean(feature_values);
        sigma = std(feature_values);
    
        normalized = (feature_values - mu) / sigma;
    
        data_norm(1,1,feature,:) = reshape(normalized, [1 1 1 num_jets]);
    end

    input_features = data_norm;

end






