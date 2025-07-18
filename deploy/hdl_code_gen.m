load('trainedModelFin90k.mat');

% generates the deep learning processor (This generates IP core HDL)
dlhdl.buildProcessor('ProjectFolder', 'HDLProject');

% export HDL Code for the trained network
dlhdl.buildFPGAImage( ...
    'Network', trainedModelFin90k, ...
    'BitstreamName', 'cnn_bitstream', ...
    'ProjectFolder', 'HDLProject');
