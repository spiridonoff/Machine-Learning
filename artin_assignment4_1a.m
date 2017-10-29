clc
clear
load train.data
load test.data
words = textread('vocabulary.txt','%s');

s_train = sparse(train(:,1),train(:,2),train(:,3));
s_test = sparse(test(:,1),test(:,2),test(:,3));
s_total = sparse([train(:,1);test(:,1)],...
    [train(:,2);test(:,2)],[train(:,3);test(:,3)]);
%% Part 1
n_train_unique_word = size(unique(train(:,2)),1)
n_test_unique_word = size(unique(test(:,2)),1)
n_total_unique_word = size(unique([train(:,2);test(:,2)]),1)
%% Part 2
n_train_docs = size(unique(train(:,1)),1);
train_doc_length = zeros(n_train_docs,1);
for i = 1:n_train_docs
    train_doc_length(i) = sum(s_train(i,:));
end
train_mean_length = mean(train_doc_length)
train_std_length = std(train_doc_length)

n_test_docs = size(unique(test(:,1)),1);
test_doc_length = zeros(n_test_docs,1);
for i = 1:n_test_docs
    test_doc_length(i) = sum(s_test(i,:));
end
test_mean_length = mean(test_doc_length)
test_std_length = std(test_doc_length)
%% Part 3
n_test_not_train = n_total_unique_word - n_train_unique_word
%% Part 4
total_word_freq = zeros(n_total_unique_word,1);
for i = 1:n_total_unique_word
    total_word_freq(i) = sum(s_total(:,i));
end

[sorted_total_word_freq, I] = sort(total_word_freq, 'descend');
for i = 1:10
    sorted_total_word_freq(i) %frequency
    a = I(i);
    words(a)     
end
%% Part 5
min_freq_word = min(total_word_freq)
n_min_freq_word = 0;

for i= 1: n_total_unique_word
    if total_word_freq(i) == min_freq_word
        n_min_freq_word = n_min_freq_word +1;
        word = words(i);
        if startsWith(word,'od')
            word
        end
    end
end
save('4_1a_results')