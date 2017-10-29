close all
clear
clc
load data_sparse.mat

n_data = size(data_s,1);
data_ext = [data_s, ones(n_data,1)]; %x^extended

%Spliting the data to training and testing set
n_train = floor(n_data*0.6);
n_test = n_data - n_train;

x_train = data_ext(1:n_train,:);
y_train = cat_id(1:n_train,:);

x_test = data_ext(n_train+1:n_data,:);
y_test = cat_id(n_train + 1:n_data,:);

%initialization
eta = 1e-5;
lambda = 100;
theta = zeros(39,42);
t_max = 500;

f = zeros(t_max,1); %l_2 regularized objective function
CCR_train = zeros(t_max,1); %CCR of the training set
CCR_test = zeros(t_max,1); %CCR of the test set
logloss = zeros(t_max,1); %logloss

%training and testing
for iter = 1:t_max
    ewtx = exp(theta*x_train'); % wtx( i, j) = exp((w_i)^T*x_j)
    sewtx = sum(ewtx); %sum of each column of ewtx
    
    ind = sparse( 1:n_train, y_train, ones(n_train,1))';
    grad = (ewtx./sewtx - ind)*x_train;
    theta = theta - eta*grad;
    theta(39,:)=zeros(1,42); %fixing W_39 to zero
    
    NLL = sum(log(sewtx)) - sum(diag(theta*(ind*x_train)'));
    f(iter) = NLL + 0.5*lambda*sum(sum(theta.^2));
    
    [~,h_train] = max(ewtx);
    CCR_train(iter) = sum(diag(confusionmat(h_train,y_train)))/n_train;
    
    ewtx_test = exp(theta*x_test');
    sewtx_test = sum(ewtx_test);
    
    [p_test,h_test] = max(ewtx_test./sewtx_test);
    CCR_test(iter) = sum(diag(confusionmat(h_test,y_test)))/n_test;
    logloss(iter) = sum(log(max(p_test,1e-10)))/(-n_test);
end

%% Part 1
plot(f)
grid on
xlabel('Iteration')
ylabel('l_2-Regularized Objective Function')
title('Objective Function vs. Iterations')
savefig('5_1b_f.fig')
saveas(gcf,'5_1b_f.jpg')

%% Part 2
figure
plot(CCR_train)
grid on
xlabel('Iteration')
ylabel('CCR train')
title('CCR train vs. Iterations')
savefig('5_1b_CCR_train.fig')
saveas(gcf,'5_1b_CCR_train.jpg')

figure
plot(CCR_test)
grid on
xlabel('Iteration')
ylabel('CCR test')
title('CCR test vs. Iterations')
savefig('5_1b_CCR_test.fig')
saveas(gcf,'5_1b_CCR_test.jpg')

figure
plot(logloss)
grid on
xlabel('Iteration')
ylabel('logloss test')
title('logloss vs. Iterations')
savefig('5_1b_logloss.fig')
saveas(gcf,'5_1b_logloss.jpg')

%% Part 3
figure
histogram(categorical(cat_unique(h_test)))
title('Histogram of Predicted Labels for Test Data')
xlabel('Labels')
ylabel('Occurance')
savefig('5_1b_cat.fig')
saveas(gcf,'5_1b_cat.jpg')


figure
histogram(categorical(cat_unique(h_test((h_test'==y_test)))))
title('Histogram of Correctly Predicted Labels for Test Data')
xlabel('Labels')
ylabel('Occurance')
savefig('5_1b_cat_correct.fig')
saveas(gcf,'5_1b_cat_correct.jpg')

save('5_1b_results');