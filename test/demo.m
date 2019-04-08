%%%%%%%%%%%%%%%%%%%%% Numerical Demonstration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This demo orignates from 
%   Ref. Q. Zhu, C. Li,
%       Dimensionality Reduction with Input Training Neural Network
%       and Its Application in Chemical Process Modelling,
%       Chinese Journal of Chemical Engineering, 14 (2006) 597-603
% where a numerical 3-D examples is provided to validate itnn algorithm.
n = 200;
mu = 0; sigma = 0.01;
t = 2*rand(1, n) - 1;
x = 0.5*t.^2 - 2*t + 0.5 + normrnd(mu, sigma,size(t));
y = t.^2 + t + sin(pi*t) + normrnd(mu, sigma,size(t));
z = 2*t.^2 - t - 2*cos(pi*t) + normrnd(mu, sigma,size(t));

X = [x; y; z];

Ni = 1;
hiddenSize = 20;
net = itnncreate(Ni, hiddenSize);
net.trainParam.mc = 0.95;
net.trainParam.lr = 0.7;
net.trainParam.lr_inc = 1.05;
net.trainParam.lr_dec = 0.70;
net.trainParam.goal = 1e-6;
net.trainParam.epoch = 200;

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

net.performFcn = 'mse';
net.adaptFcn = 'none';
net.trainParam.showCommandLine = true;
net.trainParam.showState = false;
net.trainParam.showWindow = false;
[net, tr]= itnntrain(net, X);
XPrime = tr.train.output;

xo = X(1,:);
yo = X(2,:);
zo = X(3,:);

xr = XPrime(1,:);
yr = XPrime(2,:);
zr = XPrime(3,:);

scatter3(xo,yo,zo, 'bo');
hold on
scatter3(xr, yr,zr, 'r+');
hold off
legend('Orignal data','Reconstruction data')

