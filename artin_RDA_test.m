function [Y_predict] = RDA_test(X_test, RDAmodel, numofClass)
%
% Testing for RDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix 
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Write your code here:
[n, D] = size(X_test);
Y_predict = zeros(n,1);
beta = zeros(numofClass, D); % auxiliary variable to optimize the code
alpha = zeros(numofClass); % auxiliary variable to optimize the code

for m = 1:numofClass
    beta(m,:) = RDAmodel.Mu(m,:)/(RDAmodel.Sigmapooled);
    alpha(m) = -0.5*beta(m,:)*transpose(RDAmodel.Mu(m,:))...
            + log(RDAmodel.Pi(m));
end
for i=1:n
    h = zeros(numofClass,1); %auxiliary variable for finding the argmax
    for m = 1:numofClass
        h(m) = beta(m,:)*(transpose(X_test(i,:))) + alpha(m);
    end
    [~,I]=max(h);
    Y_predict(i)=I;
end
end