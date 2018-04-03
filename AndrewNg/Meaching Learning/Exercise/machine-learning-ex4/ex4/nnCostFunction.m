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

X = [ones(m, 1) X];

%fprintf('\nSize of X: %d\n', size(X));%5000*401
%fprintf('\nSize of y: %d\n', size(y));%5000*1


%fprintf('\nSize of Theta1: %d\n', size(Theta1));%25*401
%fprintf('\nSize of Theta2: %d\n', size(Theta2));%10*26
%fprintf('\nSize of Theta1_grad: %d\n', size(Theta1_grad));
%fprintf('\nSize of Theta2_grad: %d\n', size(Theta2_grad));


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



a1 = X; 5000*401
z2 = a1*Theta1'; % 5000*401 * 401*25 = 5000*25
%a2 = sigmoid(z2); % 5000*25
a2 = [ones(m, 1) sigmoid(z2)]; % 5000*26
%c = zeros(size(a2, 1), 1);
z3 = a2*Theta2'; % 5000*10
a3 = sigmoid(z3); % 5000*10

yk = zeros(m, num_labels); % 5000*10, y=5000*1!!!!!

for i = 1:m
    yk(i, y(i)) = 1; % y存的是5000个1-10，这个循环是把y转成1和0
end

%J = (1/m) * sum(sum( -yk .* log(a3) - (1 - yk) .* log(1-a3) ));   % + lambda/(2*m) * sum(theta_1 .^ 2);


J = (1/m) * sum(sum( -yk .* log(a3) - (1 - yk) .* log(1-a3) )) + lambda/(2*m) * ( sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));

%Delta1 = zeros(size(Theta1));% 25
%Delta2 = zeros(size(Theta2));% 10%

%for t = 1:m
%delta_3 = a3(t,:) - yk(t,:); % 1*10 -1*10 = 1*10
%delta_2 = (delta_3 * Theta2)(2:end).* sigmoidGradient(z2(t,:)); % 1*10 * 10*(26-1) .* 1*25 = 1*25%

%Delta2 += a2(t,:)' * delta_3; %  26*1 * 1*10 ?
%Delta1 += a1(t,:) * delta_2 ; %  1*401 * 1*25?
%end%

%Theta2_grad = Delta2/m;  
%Theta1_grad = Delta1/m; %

%Theta2_grad(:,2:end) = Theta2_grad(:,2:end) .+ lambda * Theta2(:,2:end) / m;  
%Theta1_grad(:,2:end) = Theta1_grad(:,2:end) .+ lambda * Theta1(:,2:end) / m;  

for row = 1:m
    a1 = [X(row,:)]';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    z2 = [1; z2];
    delta3 = a3 - yk'(:, row);
    delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);
    delta2 = delta2(2:end);

    Theta1_grad = Theta1_grad + delta2 * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';

end

Theta1_grad = Theta1_grad ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ...
        + (lambda/m) * Theta1(:, 2:end);
Theta2_grad = Theta2_grad ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ...
        + (lambda/m) * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
