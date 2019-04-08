function z = sinc2d(x, y)
%SINC2D a 2-D benchmark function


%   Date: Oct. 06, 2018
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)



r = sqrt(x.^2 + y.^2) + eps;
z =  sin(r)./r;


