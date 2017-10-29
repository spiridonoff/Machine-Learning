function [RDAmodel]= RDA_train(X_train, Y_train,gamma, numofClass)
%
% Training RDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes 
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Write your code here:
[n,D] = size(X_train);
RDAmodel.Mu = zeros( numofClass , D);
RDAmodel.Sigmapooled = zeros( D, D);
RDAmodel.Pi = zeros( numofClass, 1);
for m = 1:numofClass
    ky = find( Y_train==m ); % the indices of data-points with label m.
    ny = size(ky, 1); %# of points with label m.
    
    RDAmodel.Pi(m) = ny/n; % prior probability of class m
    X = zeros( D, ny); %class-m Feature Matrix
    for i = 1:ny
        X(:, i) = X_train( ky(i),:);
    end
    RDAmodel.Mu( m,:) = X*ones( ny, 1)/ny; %class-m Empirical Feature Mean
    X_c = X*(eye(ny) - ones(ny)/ny); %class-m Centered Feature Matrix
    RDAmodel.Sigmapooled = RDAmodel.Sigmapooled + ...
        X_c*transpose(X_c)/n; % Empirical Feature Cov. Matrix
end
RDAmodel.Sigmapooled = gamma*diag(diag(RDAmodel.Sigmapooled)) + ...
    (1-gamma)*RDAmodel.Sigmapooled;
end