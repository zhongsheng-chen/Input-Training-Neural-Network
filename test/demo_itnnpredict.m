%%%%%%%%%%%%%%%%%%%%% Numerical Demonstration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is used to validate itnnperdict(). It is worthing to note
% that itnnperdict() have similay behaviors with innngeneralize(). Both
% itnnpredict() and innngeneralize() can be used to make prediction for
% ITNN when signals to be reconstructed are given.

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
net.trainParam.showCommandLine = false;
net.trainParam.showWindow = false;
[net, tr]= itnntrain(net, X);

trainInd = tr.trainInd;
valInd = tr.valInd;
testInd = tr.testInd;

traindata = X(:, trainInd);
valdata = X(:, valInd);
testdata = X(:, testInd);

[componentOfTrain, TrainRecordOnPredict] = itnnpredict(net, traindata);
[componentOfVal, ValRecordOnPredict] = itnnpredict(net, valdata);
[componentOfTest, TestRecordOnPredict] = itnnpredict(net, testdata);

trainXPrime = TrainRecordOnPredict.output;
valXPrime = ValRecordOnPredict.output;
testXPrime = TestRecordOnPredict.output;

trainFigure = figure('Name', 'Predict Principle Component for Trainning Data');
view_reconstruction(trainFigure, X, trainXPrime);

valFigure = figure('Name', 'Predict Principle Component for Validation Data');
view_reconstruction(valFigure, X, valXPrime);

testFigure = figure('Name', 'Predict Principle Component for Testing Data');
view_reconstruction(testFigure, X, testXPrime);


function fig = view_reconstruction(fig, X, XPrime)
xo = X(1,:);
yo = X(2,:);
zo = X(3,:);

xr = XPrime(1,:);
yr = XPrime(2,:);
zr = XPrime(3,:);

figure(fig)
scatter3(xo,yo,zo, 'bo');
hold on
scatter3(xr, yr,zr, 'r+');
hold off
legend('Orignal data','Reconstruction data')
end

