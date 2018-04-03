function J = costFunctionJ(X, y, theta) 
% X is "design matirx" contain training exapmle.
% y is the class labels

m = size(X,1); % # of rows
predictions = X*theta; % h(theta)x(i)
squareErrors = (predictions - y).^2; % (h(theta)x(i)-y)^2

J = (1/(2*m))*sum(squareErrors);
