function [Y_predict] = QDA_test(X_test, QDAmodel, numofClass)
%
% Testing for QDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% QDAmodel: the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance
% matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% 
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Write your code here:

n = size(X_test,1); %number of test data
Y_predict = zeros(n,1);

for i=1:n
    h = zeros(numofClass,1); %auxiliary variable for finding the argmin
    for m = 1:numofClass
        h(m) = 0.5*(X_test(i,:)-QDAmodel.Mu(m,:))*((QDAmodel.Sigma(:,:,m))\...
            (transpose(X_test(i,:)-QDAmodel.Mu(m,:))))...
            + 0.5*log(det(QDAmodel.Sigma(:,:,m))) - log(QDAmodel.Pi(m));
    end
    [~,I]=min(h);
    Y_predict(i)=I;
end
end
