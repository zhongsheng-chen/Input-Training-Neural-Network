function [P, tr] = itnnpredict(net, target)
%ITNNPREDICT Predict inputs when giving targets.

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)


global TS
if ~isequal(size(target, 1), net.numOutput)
    error('ITNN:itnnpredict:misMatch','Dimension does not match.')
end

[net, data] = itnn_configure(net, target);
[net, tr, data] = itnn_prepare(net, data);
X = data.X;
T = data.T;
pdX = data.pdX;
pdW = data.pdW;
pdb = data.pdb;
option.adjustMethod = 'fixed';
option.phase = 'prediction';
[net, P, out, errors, gradient, cpuTime, maxit, stop] = itnniter(net, X, T, pdX, pdW, pdb, option);

target = postprocess(net.processFcn, T, TS);
output = postprocess(net.processFcn, out, TS);
perf = mse(target, output);

tr.princom = P;
tr.output = output;
tr.error = errors;
tr.gradient = gradient;
tr.cpuTime = cpuTime;
tr.performance = perf;
tr.maxIteration = maxit;
tr.stop = stop;
end

function [net, data] = itnn_configure(net, target)
% Preprocess target
global TS
target = preprocess(net.processFcn, target, TS);
[No, Q] = size(target);
data.No = No;
data.Ni = net.numInput;
data.T = target;
data.Q = Q;
end

function [net, tr, data] = itnn_prepare(net, data)
% Prepare X, previous dx and previous dW  
lb = -1;
ub = 1;
Nl = net.numLayer;
for i = 1 : Nl
        pdW{i} = zeros(size(net.weight{i})); % previous dW.
        pdb{i} = zeros(size(net.bias{i})); % previous db.
end

net.layer{1}.rx = net.trainParam.lr;
for i = 1 : Nl
    net.layer{i}.rw = net.trainParam.lr;
    net.layer{i}.rb = net.trainParam.lr;
end

Ni = data.Ni;
Q = data.Q;

X = unifrnd(lb, ub, Ni, Q);
pdX = zeros(size(X));

data.X = X;
data.pdX = pdX;
data.pdW = pdW;
data.pdb = pdb;

tr.princom = [];
tr.output = [];
tr.error = [];
tr.gradient = [];
tr.cpuTime = [];
tr.performance = [];
tr.maxIteration = [];
tr.stop = [];
end


