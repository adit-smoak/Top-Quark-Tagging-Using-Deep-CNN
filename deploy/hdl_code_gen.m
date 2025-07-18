load('trainedModelFin90k.mat');

% generates the deep learning processor (This generates IP core HDL)
dlhdl.buildProcessor('ProjectFolder', 'HDLProject');

% Export HDL Code for your trained network
dlhdl.buildFPGAImage( ...
    'Network', trainedModelFin90k, ...
    'BitstreamName', 'my_cnn_bitstream', ...
    'ProjectFolder', 'HDLProject');
