function [net, P, output, error, gradient, cpuTime, maxit, stop] = itnniter(net, X, T, pdX, pdW, pdb, option)
%ITNNITER perform iteration for  dX, dW, db and record error of ITNN.

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)


goal  = net.trainParam.goal;
epoch = net.trainParam.epoch;
adjustMethod = option.adjustMethod;

efig = [];
if net.trainParam.showWindow
    efig = figure();
end
sfig = [];
if net.trainParam.showState
    sfig = figure();
end

for i = 1:epoch
    tic;
    [error(i), outLayer, E] = itnnfeedforward(net, X, T);
    [net, gradient(i), d] = itnnbackpropagation (net, E, outLayer);
    [net, X, dX, dW, db] = itnnadjust(net, X, outLayer, d, pdX, pdW, pdb, adjustMethod);
    net = itnnadapt(net, i, error, dX, pdX, dW, pdW, db, pdb);
    
    pdX = dX;
    pdW = dW;
    pdb = db;
    
    % Time interval between each iteration.
    cpuTime(i) = toc;
   
    if net.trainParam.showCommandLine
        message = sprintf('%s erros =  %3.6f and gradients %3.6f at %d epoch.\n', option.phase, error(i), gradient(i), i);
        fprintf(1, message);
    end
    
    if net.trainParam.showWindow
        option.plot.lineName = 'Erros';
        efig = itnnupdatefigure(efig, i, error, option);
    end
    
    if net.trainParam.showState
        option.plot.lineName = 'Gradients';
        sfig = itnnupdatefigure(sfig, i, gradient, option);
    end
   
    if error(i) < goal
        stop = 'goal';
        break;
    end
    stop = 'maxiteration';
end %  for  i = 1:epoch
P = X;
maxit = i;
output = outLayer{end};
end

