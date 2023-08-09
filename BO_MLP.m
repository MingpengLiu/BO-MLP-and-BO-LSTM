close all;               
clear;                   
clc;                     

%% Specify used dataset 
num_start = 1;           % the number of the first used data profile
num = 64;                % the number of the last  used data profile 
N_oneCPT = 100;          % number of points in one curve

%% Import original data and process them
data = ImportData(num_start, num, N_oneCPT); %% Process raw qc profiles

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

%% Create the function to be optimized
ObjFcn = @CostFunction;

%%  Parameter range of Bayesian optimization
optimVars = [
    optimizableVariable('NumOfUnits_1', [1, 15], 'Type', 'integer')              % number of nodes in hidden layer
    optimizableVariable('NumOfUnits_2', [1, 15], 'Type', 'integer')   
    optimizableVariable('InitialLearnRate', [1e-5, 0.01], 'Transform', 'log')];       % initial learning rate

BayesObject = bayesopt(ObjFcn, optimVars, ...    % Optimization function and parameter range 
        'MaxTime', Inf, ...                      % Optimization time: non-limited 
        'IsObjectiveDeterministic', false, ...
        'MaxObjectiveEvaluations', 10, ...       % Maximum iteration cycle
        'Verbose', 1, ...                        % Display the optimization process
        'UseParallel', false);
NumOfUnits_1     = BayesObject.XAtMinEstimatedObjective.NumOfUnits_1;       % Best number of nodes in hidden layer
NumOfUnits_2     = BayesObject.XAtMinEstimatedObjective.NumOfUnits_2;       % Best number of nodes in hidden layer
InitialLearnRate = BayesObject.XAtMinEstimatedObjective.InitialLearnRate; % Best initial learning rate

%% MPL
net = newff(xTrain, yTrain, [NumOfUnits_1,NumOfUnits_2],{'tansig', 'tansig', 'purelin'});

net.trainFcn = 'trainlm';
net.trainParam.mc = 0.00081;
net.trainParam.epochs = 200;
net.trainParam.goal   = 5e-4;
net.trainParam.lr = InitialLearnRate;
net.divideParam.trainRatio = 0.85;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0;

%train the network
net = train(net, xTrain, yTrain);

% Predictions
ysim1 = sim(net, xTrain);
YSim1 = mapminmax('reverse', ysim1, ps_output);
mse1 = mse(YTrain, YSim1);
R1 = 1 - norm(YTrain -  YSim1)^2 / norm(YTrain -  mean(YTrain))^2;

ysim2 = sim(net, xTest);
YSim2 = mapminmax('reverse', ysim2, ps_output);
mse2 = mse(YTest, YSim2);
R2 = 1 - norm(YTest -  YSim2)^2 / norm(YTest -  mean(YTest ))^2;

% plot results
figure
plot(1:size(YTest, 2), YTest,'r-',1:size(YTest, 2), YSim2,'b-')
figure
plot(1:size(YTrain, 2), YTrain,'r-',1:size(YTrain, 2), YSim1,'b-')

figure
plot(YTrain, YSim1,'*b')
set(gca, 'XLim',[0,max(YTrain)]);
set(gca, 'YLim',[0,max(YTrain)])
line([0,max(YTrain)],[0,max(YTrain)],'color','r','linewidth',2);

figure
plot(YTest, YSim2,'*b')
set(gca, 'XLim',[0,max(YTest)]);
set(gca, 'YLim',[0,max(YTest)])
line([0,max(YTest)],[0,max(YTest)],'color','r','linewidth',2);



