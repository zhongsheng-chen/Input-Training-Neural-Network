function [net, tr, data] = itnnprepare(net, data)
%ITNNPREPARE Prepare varibles pdW and pdb to trainning process.
%   [pdW, pdB] = PREPAREITNN(net) create pdW and pdb, whose size are same
%           as weights and biases.

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:Zhongsheng.chen@outlook.com)

% Initilize previous delta of weights and biases.

lb = -1;
ub = 1;

Nl = net.numLayer;
for i = 1 : Nl
        pdW{i} = zeros(size(net.weight{i})); % previous dW.
        pdb{i} = zeros(size(net.bias{i})); % previous db.
end

% Initial learning ratio.
net.layer{1}.rx = net.trainParam.lr;
for i = 1 : Nl
    net.layer{i}.rw = net.trainParam.lr;
    net.layer{i}.rb = net.trainParam.lr;
end

% Initial principle compoment matrix (X)
Ni = data.Ni;
Q = data.Q;

X = unifrnd(lb, ub, Ni, Q);
pdX = zeros(size(X));

data.X = X;
data.pdX = pdX;
data.pdW = pdW;
data.pdb = pdb;


% Initial trainning records.
tr.train.princom = [];
tr.train.output = [];
tr.train.error = [];
tr.train.gradient = [];
tr.train.cpuTime = [];
tr.train.performance = [];
tr.train.maxIteration = [];
tr.train.stop = [];
tr.val.princom = [];
tr.val.output = [];
tr.val.error = [];
tr.val.gradient = [];
tr.val.cpuTime = [];
tr.val.performance = [];
tr.val.maxIteration = [];
tr.val.stop = [];
tr.test.princom = [];
tr.test.output = [];
tr.test.error = [];
tr.test.gradient = [];
tr.test.cpuTime = [];
tr.test.performance = [];
tr.test.maxIteration = [];
tr.test.stop = [];

