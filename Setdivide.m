function [TrainingSet, TestSet] = Setdivide(Num_of_testing, data, N_oneCPT)

num       = size(data,1)/N_oneCPT;
num_start = evalin("base",'num_start');
total_test_num = size(Num_of_testing,2);
Num_of_testing = sort(Num_of_testing);

%% Divide test set
TestSet = [];
for i = 1:1:total_test_num
    n_th = Num_of_testing(i)-num_start+1;
    Set = data((((n_th-1)*N_oneCPT)+1):(n_th)*N_oneCPT,:);
    TestSet = [TestSet; Set];
end

%% Divide training set
Num_of_testing_add = [num_start-1,Num_of_testing,num+1];
TrainingSet = [];
for i = 2:1:total_test_num+2
    
    Set_train = data((Num_of_testing_add(i-1)-num_start+1)*N_oneCPT+1:((Num_of_testing_add(i)-1)-num_start+1)*N_oneCPT,:);

    TrainingSet = [TrainingSet;Set_train];
end

end