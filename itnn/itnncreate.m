function net = itnncreate(Ni, hiddenSize)
%ITNNCREATE Create a ITNN with defalult parameters. CREATEITNN does not 
%       initial weights(W), biases(b) and principle component matrix

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

net.numInput = Ni; % Number of inputs.At the same time, Ni indicates number of principle component.
net.numOutput = 0;    % Number of targets.

% Number of layers ( include hidden layer and output layer).
net.numLayer = size(hiddenSize, 2) + 1;

hiddenLayerSize = [hiddenSize, net.numOutput];
for i = 1 : net.numLayer
    if i == net.numLayer
        net.layer{i}.transferFcn = 'purelin';
    else
        net.layer{i}.transferFcn = 'logsig';
    end
    net.layer{i}.size = hiddenLayerSize(i);
end

% Normalization
net.processFcn  = 'mapminmax';
net.processParam.ymax =  1;
net.processParam.ymin = -1;

% Data division
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

net.trainParam.mc = 0.90;
net.trainParam.lr = 0.01;
net.trainParam.lr_inc = 1.05;
net.trainParam.lr_dec = 0.70;
net.trainParam.goal = 1e-6;
net.trainParam.epoch = 200;

net.performFcn = 'mse';
net.adaptFcn = 'none';

% net.adaptation.enable = false;

net.weight = [];           % Weights
net.bias = [];           % Biases

% Display
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;




