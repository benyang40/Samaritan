%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

max_iter = 1000;

data = load('../training_data.txt');
X = data(:, [1:7]); y = data(:, 11);

fprintf(['Plotting data...\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('HT Home Goals')
ylabel('HT Away Goals')

% Specified in plot order
legend('Home win', 'Draw', 'Away win')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%{
%%%%% finding correct lambda  %%%%%%%%%%%%%%%%%%%%%

% load validation data
data_val = load('../validation_data.txt');
X_val = data_val(:, [1:7]); y_val = data_val(:, 11);

[lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, X_val, y_val);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
    fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%}

%%%%% test data %%%%%%%%%%%%%%%%%%%%%
data_test = load('../test_data.txt');
X_test = data_test(:, [1:7]); y_test = data_test(:, 11);

%%%%%%%%%% train with lambda
lambda_test = 1;
[Theta1, Theta2] = train(X_test, y_test, lambda_test, max_iter);


%%%%%%%%%% predict with Theta1 and Theta2
[h, h2, pred] = predict(Theta1, Theta2, X_test);


fprintf('\nOverall Test Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);




