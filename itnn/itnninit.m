function net = itnninit(net)
%ITNNINIT Initialize weights, biases and principle compoment matrix.

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

lb = -1;
ub = 1;
Nl  = net.numLayer;
for i = 1 : Nl
    if i == 1
        net.weight{i} = unifrnd(lb, ub, net.layer{i}.size, net.numInput);
    else
        net.weight{i} = unifrnd(lb, ub, net.layer{i}.size, net.layer{i - 1}.size);
    end
    net.bias{i} = unifrnd(lb, ub, net.layer{i}.size, 1);
end
end

