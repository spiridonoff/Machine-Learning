function [QDAmodel]= QDA_train(X_train, Y_train, numofClass)
%
% Training QDA
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
% QDAmodel : the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Write your code here:
[n,D] = size(X_train); % n:# of training points, D:# of features
QDAmodel.Mu = zeros( numofClass , D);
QDAmodel.Sigma = zeros( D, D, numofClass);
QDAmodel.Pi = zeros( numofClass, 1);

for m = 1:numofClass
    ky = find( Y_train==m ); % the indices of data-points with label m.
    ny = size(ky,1); %# of points with label m.
    
    QDAmodel.Pi(m) = ny/n; % prior probability of class m
    
    X = zeros( D, ny); %class-m Feature Matrix
    for i = 1:ny
        X(:, i) = X_train( ky(i),:);
    end
    
    QDAmodel.Mu( m,:) = X*ones( ny, 1)/ny; %class-m Empirical Feature Mean
        
    X_c = X*(eye(ny) - ones(ny)/ny); %class-m Centered Feature Matrix
    QDAmodel.Sigma(:,:, m) = X_c*transpose(X_c)/ny; %class-m Empirical Feature Cov. Matrix
end
end