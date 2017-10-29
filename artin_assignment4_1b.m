clc
clear

load train.label
load test.label
train_label = train;
test_label = test;
load 4_1a_results.mat
%% Train
bag_train = sparse(train_label(train(:,1)),train(:,2),train(:,3),...
    20,n_total_unique_word);

n_word_class = zeros(20,1); % number of words in each class
for m = 1:20
    n_word_class(m) = sum(bag_train(m,:));
end
beta = sparse(train_label(train(:,1)),train(:,2),train(:,3)./...
    n_word_class(train_label(train(:,1))),20,n_total_unique_word);
beta_log = log(beta);

py = zeros(20,1); %prior probability of each class
for i = 1:size(train_label,1)
    py(train_label(i)) = py(train_label(i)) + 1/n_train_docs;
end
py_log = log(py);
%% Part 1
plot(py,'-o');
grid on
title('Prior Probability of Each Class');
xlabel('Class Label');
ylabel('Prior Probability');
%% Part 2
n_zeros_beta = size(beta,1)*size(beta,2) - nnz(beta)
%% Test
MLE = zeros(n_test_docs,1); %maximum liklehood estimate
n_zero_MLE = 0; %NO of test docs with zero likelihood
MLE_class = s_test*beta_log' + ones(n_test_docs,1)*py_log';
for j = 1:n_test_docs
    LE_log = MLE_class(j,:);
    if max(LE_log) == -Inf
        n_zero_MLE = n_zero_MLE + 1;
    end
    [LE_sorted,I] = sort(LE_log,'descend');
    MLE(j) = I(1);
    for i = 1:19
        if LE_sorted(i+1) == LE_sorted(i)
            if py(I(i+1)) > py(I(i))
                MLE(j) = I(i+1);
            end
        else
            break
        end
    end
end
%% Parts 3, 4 and 5
n_zero_MLE
CM = confusionmat(MLE, test_label)
CCR = sum(diag(CM))/n_test_docs
save('4_1b_results')