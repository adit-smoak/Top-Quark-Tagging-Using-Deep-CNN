function visualize_averaged(img_tensor, G)
    img_tensor_background_all = sum(img_tensor(:, :, 1 : G{1, 2}), 3) / 1000;
    img_tensor_top_all = sum(img_tensor(:, :, G{1, 2} + 1 : end), 3) / 1000;
    figure;
    subplot(1, 2, 1);
    imshow(img_tensor_background_all);
    title('Background Jets');
    subplot(1, 2, 2);
    imshow(img_tensor_top_all);
    title('Top Jets');