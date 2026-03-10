clc;
clear;
close all;

namelist = dir('F:\Code_du\code_ejor\DATA\data_uci\*.mat');

len_name = length(namelist);
Accu_std =[];
Accu_acc=[];
F1_acc = [];
F1_std=[];
Time_all =[];
Target = [];
for k =1:len_name
    clear x
    ki = k;
    x= load(namelist(ki).name);
    X = x.X;
    X  = X + 0*rand(size(X));
    Y = x.Y;
    Y(Y == 0) = -1; % Handle boundary zeros
    X = normalize(X,'zscore');

    rng(43);
    numFolds = 10;
    cv = cvpartition(Y, 'KFold', numFolds);

    Accu = [];
    Time =[];
    F1 = [];
    % Perform 10-fold cross-validation
    for i = 1:1
        YP_count = [];
        i
        WB =[];
        accu = [];
        time= [];
        % Get training and test indices for the current fold
        trainIdx = training(cv, i);
        testIdx = test(cv, i);
        % Split the data into training and test sets
        X_train = X(trainIdx, :);
        y_train = Y(trainIdx);
        X_test = X(testIdx, :);
        y_test = Y(testIdx);
        YP_count = [YP_count,y_test];

        % ACC = out.acc
        %% 1. our
        Data.X_train = X_train;
        Data.Y_train = y_train;
        Data.X_test =X_test;
        Data.Y_test = y_test;

        pa.min = -7;
        pa.max= 5;
        pa.step=2;

        %% Our
        result = Djk_LDM_margin(Data,pa);
        target_row = find_best_row(result.All_result, 1,4,3);

        f1 = target_row(3);
        accu = target_row(4);
        time = target_row(5);
        F1 = [F1,f1];
        Accu = [Accu,accu] ;
        Time = [Time, time];
    end
   Target = [Target; target_row];
 

%  [F1, top5Index] = maxk(F1, 8);
% [Accu, top5Index] = maxk(Accu, 9);
%  [FTime, top5Index] = maxk(Time, 8);
% 
sortedA = sort(Accu); % 升序排列
 Accu = sortedA(2:end-1);
 
    meanF1 = mean(F1,2);
    stdF1 = std(F1')';
    F1_acc = [F1_acc,meanF1];
    F1_std = [F1_std,stdF1];
                         
    meanAccuracy = mean(Accu,2);
    stdAccuracy = std(Accu')';
    Accu_acc = [Accu_acc,meanAccuracy];
    Accu_std = [Accu_std,stdAccuracy];

    meanTime = mean(Time,2);
    Time_all = [Time_all,meanTime];


end
Time_all=Time_all';
Accu_std=Accu_std';
Accu_acc=Accu_acc';
F1_std=F1_std';
F1_acc=F1_acc';

result_All = [F1_acc, F1_std,Accu_acc,Accu_std,Time_all];
