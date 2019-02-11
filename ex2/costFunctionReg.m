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

H = sigmoid(X*theta);
inside1 = -y.*log(H);
inside2 = (1-y).*log(1 - H); 
sumregulation = sum(theta(2:size(theta,1)).^2);
J = (1/m)*sum(inside1 - inside2)+(lambda/(2*m))*sumregulation;

theta(1) = 0;
prediction = H - y;
grad = (1/m) * sum(prediction.*X)+(lambda/m)*theta';
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
