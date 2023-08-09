function valError = CostFunction(optVars)

%%  Get the data 
xTrain = evalin('base','xTrain');
yTrain = evalin('base','yTrain');
YTrain = evalin('base','YTrain');
ps_output = evalin('base','ps_output');

%%  The number of input and output features 
inputSize    = size(xTrain, 1);
numResponses = size(yTrain, 1);

%%  LSTM network
opt.layers = [ ...
    sequenceInputLayer(inputSize)       % input layer

    lstmLayer(optVars.NumOfUnits_1)       % LSTM layer/hidden layer
    reluLayer                           % Relu activatation layer

    fullyConnectedLayer(numResponses)   % fullly connected layer

    regressionLayer];                   % regression layer  

%%  Building network parameters 
opt.options = trainingOptions('adam', ...             % optimization algorithm Adam
    'MaxEpochs', 300, ...                             % Maximum training cycles
    'MiniBatchSize', 100, ...                         % Minimum batch size
    'GradientThreshold', 1, ...                       % The threshold of gradient
    'InitialLearnRate', optVars.InitialLearnRate, ... % Initial learning rate
    'LearnRateSchedule', 'piecewise', ...             % Adjust the learning rate
    'LearnRateDropPeriod', 450, ...                   % Adjust the learning rate after 450 training 
    'LearnRateDropFactor',0.2, ...                    % Adjust factor of learning rate 
    'L2Regularization', optVars.L2Regularization, ... % Regularzation parameter: L2 in this model
    'ExecutionEnvironment', 'auto',...                 % Training environment %'WorkerLoad', 'double', 
    'Verbose', 0, ...                                 % Close the optimizing curve
    'Plots', 'none');                                 % Do not plot the curve

%%  Train the network
net = trainNetwork(xTrain, yTrain, opt.layers, opt.options);

ysim1 = predict(net, xTrain);                                   % Prediction output on training set            
YSim1 = mapminmax('reverse', ysim1, ps_output);
%%  Calculate the error
valError = mse(YSim1, YTrain);

end
