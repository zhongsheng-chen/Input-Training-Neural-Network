function T = postprocess(processFcn, TN, TS)
%POSTPROCESS reverse outputs of the network.
%   T = POSTPROCESS(fcn, output, TS) reverse outputs of the network 
%       according to target normalization setting (TS);

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

switch lower(processFcn)
    case {'mapminmax', 'minmax'}
        T = mapminmax('reverse', TN, TS);
    case {'mapstd', 'std'}
        T = mapstd('reverse', TN, TS);
end



