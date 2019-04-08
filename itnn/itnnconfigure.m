function [net, data] = itnnconfigure(net, target)
%ITNNCONFIGURE Configure networks inputs to match targets.

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

[No, Q] = size(target);
Ni = net.numInput;
net.numOutput = No;
net.layer{end}.size = No; 

% normalize inputs and targets.
processFcn = net.processFcn;
processParam = net.processParam;

switch lower(processFcn)
    case {'mapminmax', 'minmax'}
        [T, TS] = mapminmax(target, processParam);
    case {'mapstd', 'std'}
        [T, TS] = mapstd(target, processParam);
end
data.T = T;
data.TS = TS;
data.No = No;
data.Ni = Ni;
data.Q = Q;
[trainInd, valInd, testInd] = feval(net.divideFcn, Q, net.divideParam);
data.trainInd = trainInd;
data.valInd = valInd;
data.testInd = testInd;
end
