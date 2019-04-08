function PN = preprocess(processFcn, P, PS)
%PREPROCESS Return normalized matrix of a given matrix. Atrributes order
%       in column in the matrix.

%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

switch lower(processFcn)
    case {'mapminmax', 'minmax'}
        PN = mapminmax('apply', P, PS);
    case {'mapstd', 'std'}
        PN = mapstd('apply', P, PS);
end
