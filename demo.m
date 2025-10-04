clc;
clear;
close all;

aa =2;%3.1
% 设置随机种子以确保可复现性
rng(123);
% 每个类别的样本数
num_samples_pos =2000;
num_samples_neg= 2000;

% +1类别的均值和协方差矩阵
mean_pos = [0,0];
cov_pos =aa*eye(2);

% 生成+1类别的合成数据
X_pos = mvnrnd(mean_pos, cov_pos, num_samples_pos);

% -1类别的均值和协方差矩阵（均值为[0, 0]，协方差矩阵为单位矩阵）
mean_neg = [-4, -4];
cov_neg =aa*eye(2);


mean_add = [-1 -1];
add_pos =1*eye(2);
num_samples_add = 0;
add_point = mvnrnd(mean_add, add_pos, num_samples_add);
y_add = ones(num_samples_add, 1);

X_pos = [X_pos; add_point];

% 生成-1类别的合成数据
X_neg = mvnrnd(mean_neg, cov_neg, num_samples_neg);
% 合并两类别的数据
X = [X_pos; X_neg];

% 生成+1类别和-1类别的标签
y_pos = ones(num_samples_pos, 1);
y_neg = -ones(num_samples_neg, 1);

y_pos = [y_pos;y_add];
Y = [y_pos; y_neg];

pars = [];
out = Margin_LDM(X,Y,pars);



