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


hypo = X*theta;
hypo = sigmoid(hypo);
loghypo = log(hypo);
oneminuslog = log(1-hypo);
j = -1*y.*loghypo - (1-y).*oneminuslog;
c = ones(m,1);
J = j' * c;
J = J/m;
reg  = (theta((2:end),1).^2)'*ones(length(theta)-1,1) * (lambda / (2*m));
J = J+reg;

%for i=1:length(theta)
%k = hypo-y.*X(:,i);
%final = k'*ones(m,1);
%final = final/m;
%grad(i,1) = grad(i,1)+final;
%endfor

grad = X' * (hypo-y);
grad = grad/m;

for i=2:length(theta)
    grad(i,1) = grad(i,1)+((lambda*theta(i,1))/m);
endfor
% =============================================================

end
