%% Initialization
clear ; close all; clc

%%%%% test data %%%%%%%%%%%%%%%%%%%%%
data_test = load('../test_data.txt');
X_test = data_test(:, [1:7]); y_test = data_test(:, 11);

X_HTR = data_test(:, 8);
naive_pred = zeros(size(X_HTR, 1), 1);

for xi = 1:rows(X_HTR)
    if (X_HTR(xi) > 0)
        naive_pred(xi) = 1;
    elseif (X_HTR(xi) < 0)
        naive_pred(xi) = 3;
    else
        naive_pred(xi) = 2;
    endif
endfor

fprintf('\nNaive predict Accuracy: %f\n', mean(double(naive_pred == y_test)) * 100);


%%%%%%%%%% load theta from file
load('../config/theta.mat');


%%%%%%%%%% predict with Theta1 and Theta2
[h, h2, pred] = predict(Theta1, Theta2, X_test);



%%%%%%%%%%%%% calculate accuracy

accuracy = 0.0;
num_pred = 0;
num_correct = 0;

for i = 1:rows(y_test)
    if(h(i) >= 0.3 && h(i) < 0.4)
        num_pred++;
        if(pred(i) == y_test(i))
            num_correct++;
        endif
    endif
endfor

fprintf('\n--[0.3, 0.4)\nTest Set Accuracy: %d/%d\n', num_correct, num_pred);
fprintf('Test Set Accuracy: %f\n', (num_correct / num_pred)  * 100);

%%%%%%%%%%%%%


accuracy = 0.0;
num_pred = 0;
num_correct = 0;

for i = 1:rows(y_test)
    if(h(i) >= 0.4 && h(i) < 0.5)
        num_pred++;
        if(pred(i) == y_test(i))
            num_correct++;
        endif
    endif
endfor

fprintf('\n--[0.4, 0.5)\nTest Set Accuracy: %d/%d\n', num_correct, num_pred);
fprintf('Test Set Accuracy: %f\n', (num_correct / num_pred)  * 100);

%%%%%%%%%%%%%

accuracy = 0.0;
num_pred = 0;
num_correct = 0;

for i = 1:rows(y_test)
    if(h(i) >= 0.5 && h(i) <0.6)
        num_pred++;
        if(pred(i) == y_test(i))
            num_correct++;
        endif
    endif
endfor

fprintf('\n--[0.5, 0.6)\nTest Set Accuracy: %d/%d\n', num_correct, num_pred);
fprintf('Test Set Accuracy: %f\n', (num_correct / num_pred)  * 100);

%%%%%%%%%%%%%

accuracy = 0.0;
num_pred = 0;
num_correct = 0;

for i = 1:rows(y_test)
    if(h(i) >= 0.6 && h(i) < 0.7)
        num_pred++;
        if(pred(i) == y_test(i))
            num_correct++;
        endif
    endif
endfor

fprintf('\n--[0.6, 0.7)\nTest Set Accuracy: %d/%d\n', num_correct, num_pred);
fprintf('Test Set Accuracy: %f\n', (num_correct / num_pred)  * 100);

%%%%%%%%%%%%%

accuracy = 0.0;
num_pred = 0;
num_correct = 0;

for i = 1:rows(y_test)
    if(h(i) >= 0.7 && h(i) < 0.8)
        num_pred++;
        if(pred(i) == y_test(i))
            num_correct++;
        endif
    endif
endfor

fprintf('\n--[0.7, 0.8)\nTest Set Accuracy: %d/%d\n', num_correct, num_pred);
fprintf('Test Set Accuracy: %f\n', (num_correct / num_pred)  * 100);

%%%%%%%%%%%%%

accuracy = 0.0;
num_pred = 0;
num_correct = 0;

for i = 1:rows(y_test)
    if(h(i) >= 0.8 && h(i) < 0.9)
        num_pred++;
        if(pred(i) == y_test(i))
            num_correct++;
        endif
    endif
endfor

fprintf('\n--[0.8, 0.9)\nTest Set Accuracy: %d/%d\n', num_correct, num_pred);
fprintf('Test Set Accuracy: %f\n', (num_correct / num_pred)  * 100);


fprintf('\nOverall Test Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);



