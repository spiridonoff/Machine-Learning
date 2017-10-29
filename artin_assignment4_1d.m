clc
clear
load 4_1c_results.mat

%% Train
bag_train = sparse(train_label(train(:,1)),train(:,2),train(:,3),...
    20,n_total_unique_word);

n_word_class = zeros(20,1); % number of words in each class
for m = 1:20
    n_word_class(m) = sum(bag_train(m,:));
end
alpha = 1/n_total_unique_word;
beta = (bag_train + alpha)./(n_word_class+1);
beta_log = log(beta);

%% Test
MAP = zeros(n_test_docs,1); %maximum aposteriori probability
MAP_class = s_test*beta_log';
for j = 1:n_test_docs
    AP_log = MAP_class(j,:);
    [AP_sorted,I] = sort(AP_log,'descend');
    MAP(j) = I(1);
    for i = 1:19
        if AP_sorted(i+1) == AP_sorted(i)
            if py(I(i+1)) > py(I(i))
                MAP(j) = I(i+1);
            end
        else
            break
        end
    end
end

%% Parts 1 and 2
CM = confusionmat(MAP, test_label)
CCR = sum(diag(CM))/n_test_docs

%% Part 3
conf = zeros(20,20); %degree of confusion for each pair
for i=1:19
    for j=i+1:20
        conf(i,j) = (CM(i,j)/sum(CM(:,j))+ CM(j,i)/sum(CM(:,i)))/2;
    end
end
[M,I]=sort(conf(:),'descend');
pairs = [rem(I(1:5),20),ceil(I(1:5)/20)] %pairs with most confusion
M(1:5) %degree of the confusion

save('4_1d_results')