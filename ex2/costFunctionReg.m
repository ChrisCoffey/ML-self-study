function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h_of_x = sigmoid(X * theta);
y_ones = ones(size(y));
regularization_expr = (lambda / (2 *m) ) * sum(theta(2:end) .^ 2);
J = ((1/m) * sum( ((y .* -1) .* log(h_of_x)) - ((y_ones - y) .* log(y_ones - h_of_x)) )) + regularization_expr;

grad = ((h_of_x - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;


% =============================================================

end
