function [net, tr] = itnntrain( net, target)
%ITNNTRAIN Train ITNN. Initially, ITNN is configured and and Initialized.
%   Then the normalized instance is fed to ITNN for trainning.

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:Zhongsheng.chen@outlook.com)

global TS



[net, data] = itnnconfigure(net, target);
net = itnninit(net);
[net, tr, data] = itnnprepare(net, data);
trainInd = data.trainInd;
valInd = data.valInd;
testInd = data.testInd;

X = data.X(:, trainInd);
T = data.T(:, trainInd);
pdX = data.pdX(:, trainInd);
pdW = data.pdW;
pdb = data.pdb;

TS = data.TS;
option.phase = 'training';
option.adjustMethod = 'update';

[net, P, out, error, gradient, cpuTime, maxit, stop] = itnniter(net, X, T, pdX, pdW, pdb, option);
target = postprocess(net.processFcn, T, TS);
output = postprocess(net.processFcn, out, TS);
perf = mse(target, output);
tr.train.princom = P;
tr.train.output = output;
tr.train.error = error;
tr.train.gradient = gradient;
tr.train.cpuTime = cpuTime;
tr.train.performance = perf;
tr.train.maxIteration = maxit;
tr.train.stop = stop;

if ~isempty(valInd)
    net.layer{1}.rx = net.trainParam.lr;
    XV = data.X(:, valInd);
    TV = data.T(:, valInd);
    pdX = data.pdX(:, valInd);
    pdW = data.pdW;
    pdb = data.pdb;
    
    option.phase = 'validation';
    option.adjustMethod = 'fixed';
    [net, PV, out, error, gradient, cpuTime, maxit, stop] = itnniter(net, XV, TV, pdX, pdW, pdb, option);
    target = postprocess(net.processFcn, TV, TS);
    output = postprocess(net.processFcn, out, TS);
    perf = mse(target, output);
    tr.val.princom = PV;
    tr.val.output = output;
    tr.val.error = error;
    tr.val.gradient = gradient;
    tr.val.cpuTime = cpuTime;
    tr.val.performance = perf;
    tr.val.maxIteration = maxit;
    tr.val.stop = stop;
end

if ~isempty(testInd)
    net.layer{1}.rx = net.trainParam.lr;
    XE = data.X(:, testInd);
    TE = data.T(:, testInd);
    pdX = data.pdX(:, testInd);
    pdW = data.pdW;
    pdb = data.pdb;
    
    option.phase = 'testing';
    option.adjustMethod = 'fixed';
    [net, PE, out, error, gradient, cpuTime, maxit, stop] = itnniter(net, XE, TE, pdX, pdW, pdb, option);
    target = postprocess(net.processFcn, TE, TS);
    output = postprocess(net.processFcn, out, TS);
    perf = mse(target, output);
    tr.test.princom = PE;
    tr.test.output = output;
    tr.test.error = error;
    tr.test.gradient = gradient;
    tr.test.cpuTime = cpuTime;
    tr.test.performance = perf;
    tr.test.maxIteration = maxit;
    tr.test.stop = stop;
end

tr.trainInd = trainInd;
tr.valInd = valInd;
tr.testInd = testInd;
end
