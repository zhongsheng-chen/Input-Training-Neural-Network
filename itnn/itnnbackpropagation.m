function [net, gradient, d] = itnnbackpropagation(net, error, outlayer)
%ITNNBACKPROPAGATION Adjust weights and biases of the net according the
%       BP learning algorithm with moment. Return gradient of output layer
%       if output transferFcn is logsig. Otherwise return gradient of last
%       hidden layer if output transferFcn is purelin.

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)


Nl =  net.numLayer;
switch net.layer{Nl}.transferFcn
    case {'logsig'}
        gradient = outlayer{Nl} .* (1 - outlayer{Nl});
        d{Nl} =  gradient .* error;
    case {'purelin', 'softmax'}
        gradient = outlayer{Nl - 1} .* (1 - outlayer{Nl - 1});
        d{Nl} = error;
end

for i = Nl - 1 : -1 : 1
    switch net.layer{i}.transferFcn
        case {'logsig'}
            grad = outlayer{i} .* (1 - outlayer{i});
        case {'tansig'}
            grad = 1 - outlayer(i) .^ 2;
        case {'tansigopt'}
            grad = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * outlayer{i} .^ 2);
    end
    d{i} = grad .* (net.weight{i + 1}' * d{i + 1});
end
% Return average gradient.
gradient = sum(sum(gradient)) / (size(gradient, 1) * size(gradient, 2));
end




