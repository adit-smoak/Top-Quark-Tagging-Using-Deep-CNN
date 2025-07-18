load('trainedModelFin90k.mat');  

%% Create Target Object (Using Ethernet or JTAG interface)
% Ethernet example:
hTarget = dlhdl.Target('Xilinx','Interface','Ethernet');
% OR, for JTAG (if you’re using that):
% hTarget = dlhdl.Target('Xilinx','Interface','JTAG');

hW = dlhdl.Workflow( ...
    'Network', trainedModelFin90k, ...   
    'Bitstream', 'zcu102_single', ...
    'Target', hTarget);

% Compile Network for FPGA
dn = hW.compile('InputFrameNumberLimit',15);

% Deploy Network to FPGA
% This step programs the FPGA and downloads the network parameters.
hW.deploy();