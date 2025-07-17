function visualize_jet(img_tensor, jet_idx)
    %img = img_tensor(:, :, jet_idx);
    %img_norm = img / max(img(:));
    %colormap_name = 'hot';
    %cmap = colormap(colormap_name);
    %rgb_img = ind2rgb(gray2ind(img_norm, size(cmap, 1)), cmap);
    %imshow(rgb_img);
    %title('False Color Jet Image');
    %xlabel('Pseudorapidity (η)');
    %ylabel('Azimuthal angle (ϕ)');
    figure;
    imagesc(img_tensor(:, :, jet_idx));
    colormap(gray);  
    title(sprintf('Average Energy per Pixel - Jet #%d', jet_idx));
    xlabel('\phi (column)');
    ylabel('\eta (row)');
    axis equal;
