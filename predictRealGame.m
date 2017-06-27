%% Initialization
clear ; close all; clc

%%%%% test data %%%%%%%%%%%%%%%%%%%%%
data_real = load('realgame.txt');
X_real = data_real(:, :);

%%%%%%%%%% load theta from file
load('theta.mat');


%%%%%%%%%% predict with Theta1 and Theta2
[h, h2, pred] = predict(Theta1, Theta2, X_real);


%%%%%%%%%% output
output = 1 ./ h2;
save('predictRealgameOutput.csv', 'output');
