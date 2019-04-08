function [net, X, dX, dW, db] = itnnadjust(net, X, outlayer, d, pdX, pdW, pdb, adjustMethod)
%ITNNADJUST Adjust X until the output approximate to target(T) .

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:Zhongsheng.chen@outlook.com)



mc = net.trainParam.mc;
rx =  net.layer{1}.rx;
dX = (net.weight{1})' * d{1};
dX = rx .* dX;
if mc > 0
    dX = mc * pdX + (1 - mc) * dX;
end
X = X + dX;

% Compute dW and db.
Nl =  net.numLayer;
Q = size(X, 2);
for i = Nl : -1 : 1
    if i == 1
        dW{i} = (d{i} * X') / Q;
    else
        dW{i} = (d{i} * outlayer{i - 1}') / Q;
    end
    db{i} = d{i} * ones(1, Q)' / Q;
end

if strcmpi(adjustMethod, 'update')
    % Adjust weights and biases.
    for i = Nl : -1 : 1
        rw = net.layer{i}.rw;
        rb = net.layer{i}.rb;
        
        dW{i} = rw .* dW{i};
        db{i} = rb .* db{i};
        
        if mc > 0
            dW{i} = mc * pdW{i} + (1 - mc) * dW{i};
            db{i} = mc * pdb{i} + (1 - mc) * db{i};
        end
        
        % adjust W and b.
        net.weight{i} = net.weight{i} + dW{i};
        net.bias{i} = net.bias{i} + db{i};
    end
end
end



