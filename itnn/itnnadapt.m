function net = itnnadapt(net, i, error, dX, pdX, dW, pdW, db, pdb)
%ITNNADAPT Adjust learning ratio on each layers adaptively.
%   net = ADAPTLR(net, dX, pdX, dW, pdW, db, pdb) takes a adaptive strategy
%       to update learn ratio of x, weights and biases, that is, if the
%       gradient descent directions ars same when continuously iterating
%       twice, the learning rate will be doubled. On the contrary, the
%       learning rate  will be halved.
%

%   Ref. Q. Zhu, C. Li,
%       Dimensionality Reduction with Input Training Neural Network
%       and Its Application in Chemical Process Modelling,
%       Chinese Journal of Chemical Engineering, 14 (2006) 597-603

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)


adaptFcn = net.adaptFcn;
Nl =  net.numLayer;
switch lower(adaptFcn)
    case {'dxpdx'}
        rx =  net.layer{1}.rx;
        lamdax = sign(dX .* pdX);
        rx = 2 .^(lamdax) .* rx;
        net.layer{1}.rx = rx;
        
        for i = 1 : Nl
            rw = net.layer{i}.rw;
            lamdaw = sign(dW{i} .* pdW{i});
            rw = 2 .^(lamdaw) .* rw;
            
            rb = net.layer{i}.rb;
            lamdab = sign(db{i} .* pdb{i});
            rb = 2 .^(lamdab) .* rb;
            
            net.layer{i}.rw = rw;
            net.layer{i}.rb = rb;
        end
    case {'annealing1'}  % Fix me
        u = net.trainParam.lr_inc;
        d = net.trainParam.lr_dec;
        
        for i = 1 : Nl
            lr = net.trainParam.lr;
            dWB = [dW{i}, db{i}];
            pdWB = [pdW{i}, pdb{i}];
            H = sign(dWB .* pdWB);
            p = find(H == 1); np =  find(H == -1);
            lr(p) = lr(p) .* u; lr(np) = lr(np) .* d;
            net.layer{i}.rx = lr;
            net.layer{i}.rw = lr;
            net.layer{i}.rb = lr;
        end
    case {'annealing2'}
        MIN = 0.1;
        MAX = 0.9;
        T = net.trainParam.epoch;
        lr = MAX - (MAX - MIN) .* i ./T;
        
        for i = 1 : Nl
            net.layer{i}.rx = lr;
            net.layer{i}.rw = lr;
            net.layer{i}.rb = lr;
        end
    case {'annealing3'}
        theta = 0.1;
        if i > 1
            lr = net.trainParam.lr;
            if error(i) < error(i - 1)
                lr = (1 + theta) .* lr;
            else
                lr = (1 - theta) .* lr;
            end
            for i = 1 : Nl
                net.layer{i}.rx = lr;
                net.layer{i}.rw = lr;
                net.layer{i}.rb = lr;
            end
        end
    case {'none'}
        return
    otherwise
        error('Unknown adaptive method')
end




