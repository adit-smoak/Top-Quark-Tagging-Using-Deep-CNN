function aligned_img = align_img(image, n, eta, phi, Xedges, Yedges)
    [cx, cy] = deal(ceil(size(image, 2)/2), ceil(size(image, 1)/2));

    row_idx = discretize(eta(:, n), Xedges);
    col_idx = discretize(phi(:, n), Yedges);

    sec_row = row_idx(2);
    sec_col = col_idx(2);
    if isnan(sec_col) || isnan(sec_row)
        aligned_img = image;
        return
    end
    % fprintf('row: %f\n', sec_row);
    % fprintf('col: %f\n', sec_col);
    val = image(sec_row, sec_col);

    dx = sec_col - cx;
    dy = -(sec_row - cy);

    theta = atan2(dy, dx);
    theta_deg = rad2deg(theta);

    % angle_adjust = 0;
    if theta_deg > 45 && theta_deg <= 135
        image_coarse = rot90(image, -1);  % 90° CW
        % angle_adjust = -90;
    elseif theta_deg > 135 || theta_deg <= -135
        image_coarse = rot90(image, -2);  % 180° CW
        % angle_adjust = -180;
    elseif theta_deg > -135 && theta_deg <= -45
        image_coarse = rot90(image, -3);  % 270° CW
        % angle_adjust = -270;
    else
        image_coarse = image;
    end

    [row2_all, col2_all] = find(image_coarse == val);

    % Use the first match only (just one pixel)
    row2 = row2_all(1);
    col2 = col2_all(1);

    dx2 = col2 - ceil(size(image_coarse, 2)/2);
    dy2 = -(row2 - ceil(size(image_coarse, 1)/2));
    
    theta2 = atan2(dy2, dx2);
    fine_rotation = -rad2deg(theta2);
    % fprintf('theta: %f\n', theta2);
    % fprintf('fine: %f\n', fine_rotation);

    aligned_img = imrotate(image_coarse, fine_rotation, 'nearest', 'crop');
end
