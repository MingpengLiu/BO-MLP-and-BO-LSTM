function valError = CostFunction(optVars)

%%  Get the data 
xTrain = evalin('base','xTrain');
yTrain = evalin('base','yTrain');
YTrain  = evalin('base','YTrain');
ps_output = evalin('base','ps_output');

%%  network
net = newff(xTrain, yTrain, [optVars.NumOfUnits_1, optVars.NumOfUnits_2], {'tansig', 'tansig', 'purelin'});

%%  Building network parameters 
net.trainFcn = 'trainlm';
net.trainParam.mc = 0.00081;
net.trainParam.epochs = 200;
net.trainParam.goal   = 5e-4;
net.trainParam.lr = optVars.InitialLearnRate;
net.divideParam.trainRatio = 0.85;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0;
%%  Train the network
net = train(net, xTrain, yTrain);

yPred = sim(net, xTrain);
YPred = mapminmax('reverse', yPred, ps_output);
%%  Calculate the error
valError = double(mse(YPred, YTrain));

end
