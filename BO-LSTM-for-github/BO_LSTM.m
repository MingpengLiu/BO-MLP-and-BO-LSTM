close all;               
clear;                   
clc;                     

%% Specify used dataset 
num_start = 1;           % the number of the first used data profile
num = 64;                % the number of the last  used data profile 
N_oneCPT = 100;          % number of points in one curve

%% Import original data and process them
data = ImportData(num_start, num, N_oneCPT); %% Process raw experimental

%% Training set and testing set
Num_of_testing = [4,8,15,19,24,34,38,40,43,47,51,59]; %% the number of cases used for testing set  

[TrainingSet, TestSet] = Setdivide(Num_of_testing, data, N_oneCPT); %% divide training set and testing set

XTrain = TrainingSet(:,1:end-1)';
YTrain = TrainingSet(:,end)';
M = size(XTrain,2);

XTest=TestSet(:,1:end-1)';
YTest=TestSet(:,end)';
N = size(YTest,2);

%%  Data normalization
[xTrain, ps_input] = mapminmax(XTrain, 0, 1);
xTest = mapminmax('apply', XTest, ps_input);

[yTrain, ps_output] = mapminmax(YTrain, 0, 1);
yTest = mapminmax('apply', YTest, ps_output);

outdim=1;                               % dimension of output
f_ = size(xTrain, 1);                  % dimension of input

%%  Create the function to be optimized
ObjFcn = @CostFunction;

%%  Parameter range of Bayesian optimization
optimVars = [
    optimizableVariable('NumOfUnits_1', [1, 20], 'Type', 'integer')              % number of nodes in hidden layer
    optimizableVariable('InitialLearnRate', [1e-4, 0.01], 'Transform', 'log')       % initial learning rate
    optimizableVariable('L2Regularization', [1e-5, 0.01], 'Transform', 'log')]; % Ridge regularization parameter
%%  BO-LSTM parameters
BayesObject = bayesopt(ObjFcn, optimVars, ...    % Optimization function and parameter range 
        'MaxTime', Inf, ...                      % Optimization time: non-limited 
        'IsObjectiveDeterministic', false, ...
        'MaxObjectiveEvaluations', 10, ...       % Maximum iteration cycle
        'Verbose', 1, ...                        % Display the optimization process
        'UseParallel', false);

%%  Get the optimzied parameter
NumOfUnits_1     = BayesObject.XAtMinObjective.NumOfUnits_1;       % Best number of nodes in hidden layer
InitialLearnRate = BayesObject.XAtMinObjective.InitialLearnRate; % Best initial learning rate
L2Regularization = BayesObject.XAtMinObjective.L2Regularization; % Best L2 regularization parameter
  
%%  Creat network
layers = [ ...
    sequenceInputLayer(f_)              % Input layer 
    
    lstmLayer(NumOfUnits_1)             % LSTM layer/hidden layer
    reluLayer                           % Relu activatation layer

    fullyConnectedLayer(outdim)         % fullly connected layer

    regressionLayer];                   % regression layer  

options = trainingOptions('adam', ...                 % optimization algorithm Adam
    'MaxEpochs',300, ...                             % Maximum training cycles
    'MiniBatchSize', 100, ...                % Minimum batch size
    'GradientThreshold', 1, ...                       % The threshold of gradient
    'InitialLearnRate', InitialLearnRate, ...         % Initial learning rate
    'LearnRateSchedule', 'piecewise', ...             % Adjust the learning rate
    'LearnRateDropPeriod', 450, ...                   % Adjust the learning rate after 450 training 
    'LearnRateDropFactor',0.2, ...                    % Adjust factor of learning rate
    'L2Regularization', L2Regularization, ...         % Regularzation parameter: L2 in this model
    'ExecutionEnvironment', 'auto',...                 % Training environment 
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...                                 % Close the optimizing curve
    'Plots', 'training-progress');                    % Do not plot the curve

%% Train the network
[net, info] = trainNetwork(xTrain,yTrain, layers, options);     % Training the network with training dataset

%% predictions
ysim1 = predict(net, xTrain);                                   % Prediction output on training set            
YSim1 = mapminmax('reverse', ysim1, ps_output);
mse1 = sum((YSim1 - YTrain).^2)./M;
R1 = 1 - norm(YTrain - YSim1)^2 / norm(YTrain - mean(YTrain))^2;

ysim2 = predict(net, xTest);
YSim2 = mapminmax('reverse', ysim2, ps_output);
R2 = 1 - norm(YTest -  YSim2)^2 / norm(YTest -  mean(YTest ))^2;
mse2 = mse(YSim2, YTest);

%% plot figure
figure;                 
plot(1:size(YTest,2),YTest,'r-',1:size(YTest,2),YSim2,'b-');
figure;                  
plot(1:size(YTrain,2),YTrain,'r-',1:size(YTrain,2),YSim1,'b-');
figure;                  

figure;                 
plot(YTest, YSim2,'*b')
set(gca, 'XLim',[0,max(YTest)]);
set(gca, 'YLim',[0,max(YTest)])
line([0,max(YTest)],[0,max(YTest)],'color','r','linewidth',2);

plot(YTrain, YSim1,'*b')
set(gca, 'XLim',[0,max(YTrain)]);
set(gca, 'YLim',[0,max(YTrain)])
line([0,max(YTrain)],[0,max(YTrain)],'color','r','linewidth',2);