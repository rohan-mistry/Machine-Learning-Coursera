function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
hypo = X*theta;
hypo = sigmoid(hypo);
loghypo = log(hypo);
oneminuslog = log(1-hypo);
j = -1*y.*loghypo - (1-y).*oneminuslog;
c = ones(m,1);
J = j' * c;
J = J/m;


%for i=1:length(theta)
%k = hypo-y.*X(:,i);
%final = k'*ones(m,1);
%final = final/m;
%grad(i,1) = grad(i,1)+final;
%endfor

grad = X' * (hypo-y);
grad = grad/m;












% =============================================================

end
