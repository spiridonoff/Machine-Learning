close all
clear
clc

load data_mnist_train
load data_mnist_test

n_test = 10000;
n_train = 60000;
numofClass = 10;
D = 784;

Y_predict = zeros(n_test,1);

B = zeros(1,n_train);
for i = 1:n_train
    B(i) = norm(X_train(i,:))^2;
end

parfor i = 1:n_test
    A = X_test(i,:)*transpose(X_train);
    C = B - 2*A;
    [~,I] = min(C);
    Y_predict(i) = Y_train(I);
end
CM = confusionmat(Y_test(1:n_test), Y_predict)
CCR = sum(diag(CM))/n_test

