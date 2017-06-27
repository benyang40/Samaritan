function [Theta1, Theta2] = ...
    train(X, y, lambda, max_iter)
    
    %% Setup the parameters you will use for this exercise
    input_layer_size  = 8;    % League HTHC    HTAC    B365H   B365D   B365A   HTHG    HTAG
    hidden_layer_size = 3;    % 3 hidden units
    num_labels = 3;           % 3 labels, 2: Home win, 1: Draw, 0: Away win 
    
    m = size(X, 1);
    
    fprintf('\nInitializing Neural Network Parameters ...\n')
    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    
    %% =================== Part 8: Training NN ===================
    %  You have now implemented all the code necessary to train a neural 
    %  network. To train your neural network, we will now use "fmincg", which
    %  is a function which works similarly to "fminunc". Recall that these
    %  advanced optimizers are able to train our cost functions efficiently as
    %  long as we provide them with the gradient computations.
    %
    fprintf('\nTraining Neural Network... \n')
    
    %  After you have completed the assignment, change the MaxIter to a larger
    %  value to see how more training helps.
    options = optimset('MaxIter', max_iter);
    
    fprintf("Lambda = %f\n", lambda);
    
    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda);
    
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
        
    % run 10 more times and find best nn_params with lowest cost
    for i = 1:10
	    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
	    
	    % Unroll parameters
	    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    	[i_params, i_cost] = fmincg(costFunction, initial_nn_params, options);
	    fprintf("%d run done\n", i);
    	if(i_cost < cost)
    		fprintf("%d run overriding nn_params\n", i);
    		nn_params = i_params;
    		fprintf("%f -> %f\n", cost, i_cost);
    		cost = i_cost;
    	endif
    endfor
    
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    %{
    %% ================= Part 9: Visualize Weights =================
    %  You can now "visualize" what the neural network is learning by 
    %  displaying the hidden units to see what features they are capturing in 
    %  the data.
    
    fprintf('\nVisualizing Neural Network... \n')
    
    displayData(Theta1(:, 2:end));
    
    fprintf('\nProgram paused. Press enter to continue.\n');
    pause;
    
    %% ================= Part 10: Implement Predict =================
    %  After training the neural network, we would like to use it to predict
    %  the labels. You will now implement the "predict" function to use the
    %  neural network to predict the labels of the training set. This lets
    %  you compute the training set accuracy.
    
    [h, h2, pred] = predict(Theta1, Theta2, X);
    
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
    %}

% =========================================================================

end




