function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

a2 = sigmoid(X * Theta1');

% Add ones to the a2 data matrix
m_a2 = size(a2, 1);
a2 = [ones(m_a2, 1) a2];

% Compute a3, output layer, or h(x)
a3 = sigmoid(a2 * Theta2');
m_a3 = size(a3, 1);

% Compute J
for c = 1:m_a3
    Yc = zeros(num_labels, 1);
    Yc(y(c)) = 1;
    J = J + (-Yc' * log(a3(c, :)') - (1 - Yc') * log(1 - a3(c, :)'));
endfor

% Regularized
regularized_part = 0;

for j = 1:size(Theta1, 1)
    for k = 2:size(X, 2)
        regularized_part = regularized_part + Theta1(j, k) .* Theta1(j, k);
    endfor    
endfor    

for j = 1:size(Theta2, 1)
    for k = 1:size(Theta1, 1)
        regularized_part = regularized_part + Theta2(j, k+1) .* Theta2(j, k+1);
    endfor    
endfor    

J = J ./ m_a3 + (lambda ./ (2 .* m_a3)) .* regularized_part;


% Backpropagation Gradients
DELTA1 = zeros(size(Theta1, 1), size(X, 2)); % 25*401
DELTA2 = zeros(size(Theta2, 1), size(Theta1, 1)+1); % 10*26

%%%% non-vectorized version (submitted)
%{
for t = 1:m % m=5000
    a_1 = X(t, :); % 1*401
    z_2 = a_1 * Theta1'; % 1*25
    a_2 = sigmoid(z_2); % 1*25
    a_2 = [1 a_2]; % 1*26
    z_3 = a_2 * Theta2'; % 1*10
    a_3 = sigmoid(z_3); % 1*10

    delta_3 = zeros(num_labels, 1); % 10*1
    for k = 1:num_labels
        delta_3(k) = a_3(k) - (y(t) == k);
    endfor
    
    z_2 = [0 z_2];
    delta_2 = Theta2' * delta_3 .* (sigmoidGradient(z_2))';
    
    % remove delta 0 (bias)
    delta_2 = delta_2(2:end);
    
    DELTA1 = DELTA1 + delta_2 * a_1;
    DELTA2 = DELTA2 + delta_3 * a_2;
endfor
%}

%%%% vectorized version
a_1 = X;    % 5000*401
z_2 = a_1 * Theta1';    % 5000*25
a_2 = sigmoid(z_2); % 5000*25
a_2 = [ones(m, 1) a_2]; % 5000*26
z_3 = a_2 * Theta2'; % 5000*10
a_3 = sigmoid(z_3); % 5000*10
delta_3 = zeros(m, num_labels); % 5000*10

for k = 1:num_labels
    delta_3(:, k) = a_3(:, k) - (y == k);
endfor

z_2 = [zeros(m, 1) z_2]; % 5000*26

%{
fprintf("size(z_2):\n");
fprintf("%d\n", size(z_2));
fprintf("size(sigmoidGradient(z_2)):\n");
fprintf("%d\n", size(sigmoidGradient(z_2)));
%}

delta_2 = Theta2' * delta_3' .* (sigmoidGradient(z_2))'; % 26*5000

% remove delta 0 (bias)
delta_2 = delta_2(2:end, :);   % 25*5000

%{
fprintf("size(delta_2):\n");
fprintf("%d\n", size(delta_2));
fprintf("size(a_1):\n");
fprintf("%d\n", size(a_1));
%}

DELTA1 = DELTA1 + delta_2 * a_1;

DELTA2 = DELTA2 + delta_3' * a_2;


Theta1_grad = DELTA1 ./ m;
Theta2_grad = DELTA2 ./ m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda ./ m) .* Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda ./ m) .* Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
