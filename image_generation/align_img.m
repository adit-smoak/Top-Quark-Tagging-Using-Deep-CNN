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

    % val = image(sec_row, sec_col);

    dx = sec_col - cx;
    dy = -(sec_row - cy);

    theta = atan2(dy, dx);
    theta_deg = rad2deg(theta);

    if theta_deg > 45 && theta_deg <= 135
        aligned_img = rot90(image, -1);  % 90° CW
    elseif theta_deg > 135 || theta_deg <= -135
        aligned_img = rot90(image, -2);  % 180° CW
    elseif theta_deg > -135 && theta_deg <= -45
        aligned_img = rot90(image, -3);  % 270° CW
    else
        aligned_img = image;
    end
end
