function [Output, P, net, tr] = itnngeneralize(net, target)
%ITNNGENERALIZE Generalize network of ITNN.
%   [Output, P, net, tr] = GENERALIZEITNN(net, t) takes a net 
%       trained by trainitnn to compute principle compoment matrix of a 
%       given targets.

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:Zhongsheng.chen@outlook.com)

global TS 
minargs = 1; maxargs = 2;
narginchk(minargs, maxargs);

if ~isequal(size(target, 1), net.numOutput)
    error('ITNN:generalization:misMatch', 'Dimension does not match.')
end

T = preprocess(net.processFcn, target, TS);

% Update size to match the given targets for contructing proper X, pdX, pdW, pdb.
[net, data]= itnnconfigure(net, T);
[net, tr, data] = itnnprepare(net, data);

X = data.X;
pdX = data.pdX;
pdW = data.pdW;
pdb = data.pdb;

option.phase = 'generalization';
option.adjustMethod = 'fixed';
[net, P, Out, Error, Gradient, cpuTime, maxit, stop] = itnniter(net, X, T, pdX, pdW, pdb, option);
Target = postprocess(net.processFcn, T, TS);
Output = postprocess(net.processFcn, Out, TS);
perf = mse(Target, Output);
tr.generalization.princom = P;
tr.generalization.output = Output;
tr.generalization.error = Error;
tr.generalization.gradient = Gradient;
tr.generalization.cpuTime = cpuTime;
tr.generalization.performace = perf;
tr.generalization.maxIteration = maxit;
tr.generalization.stop = stop;
end