function  [loss, outLayer, error] = itnnfeedforward(net, input, target)
%ITNNFEEDFORWARD Return output of layers according to weights and
%                   biases in forward propagation phrase and error (loss).
%   net = FEEDFORWARD(net, input, target) Feed inputs (in) to
%                           neural network (net), and calculate each
%                           outputs of each layers in turn.

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

% the number of batch samples (batch size).
Q = size(input, 2); 

% number of layers (numHiddenLayers + numOutputs).
Nl = net.numLayer;

% the output of input layer of the network is same as the input of input layer.
out = input;

% calculate output of hidden layer of the network.
for i = 1 : Nl - 1
    Wb = [net.bias{i}, net.weight{i}];
    X = [ones(1, Q); out];
    switch net.layer{i}.transferFcn
        case {'logsig'}
            out = logsig(Wb*X);
            outLayer{i} = out;
        case {'tansig'}
            out = tansig(Wb*X);
            outLayer{i} = out;
        case {'tansigopt'}
            out = tansigopt(Wb*X);
            outLayer{i} = out;
    end
end

% calculate output of output layer of the network.
Wb = [net.bias{Nl}, net.weight{Nl}];
X = [ones(1, Q); outLayer{Nl -1}];
switch net.layer{Nl}.transferFcn
    case {'logsig'}
        outLayer{Nl} = logsig(Wb*X);
    case {'purelin'}
        outLayer{Nl} = purelin(Wb*X);
    case {'softmax'}
        outLayer{Nl} = softmax(Wb*X);
end


output = outLayer{Nl};
error = target - output;

switch net.layer{Nl}.transferFcn
    case {'logsig', 'purelin'}
        loss = 1/2 * sum(sum(error .^ 2)) / Q;
    case {'softmax'}
        loss = -sum(sum(target .* log(output))) / Q;
end

