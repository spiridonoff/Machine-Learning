clc
clear
load 4_1e_results.mat

words = textread('vocabulary.txt','%s');
n_words = size(words,1);

stoplist = textread('stoplist.txt','%s');
n_stoplist = size(stoplist,1);

[nonstop,ia] = setdiff(words,stoplist,'stable');

n_nonstop = size(nonstop,1);
is_nonstop = zeros(n_words,1); %1 if it is nonstop, 0 otherwise
id_nonstop = ones(n_words,1)*(n_nonstop + 1);% new id of nonstop words
for i = 1:n_nonstop
    id_nonstop(ia(i)) = i;
    is_nonstop(ia(i)) = 1;
end

%% Part 1
n_train_unique_word_ns = size(unique(id_nonstop(train(:,2))),1)-1
n_test_unique_word_ns = size(unique(id_nonstop(test(:,2))),1)-1
n_total_unique_word_ns = size(unique(id_nonstop([train(:,2);test(:,2)])),1)-1

s_train = sparse(train(:,1),id_nonstop(train(:,2)),train(:,3));
s_test = sparse(test(:,1),id_nonstop(test(:,2)),test(:,3));
s_total = sparse([train(:,1);test(:,1)],...
    id_nonstop([train(:,2);test(:,2)]),[train(:,3);test(:,3)]);

%% Part 2
train_doc_length = zeros(n_train_docs,1);
for i = 1:n_train_docs
    train_doc_length(i) = sum(s_train(i,1:n_nonstop));
end
train_mean_length = mean(train_doc_length)
train_std_length = std(train_doc_length)

test_doc_length = zeros(n_test_docs,1);
for i = 1:n_test_docs
    test_doc_length(i) = sum(s_test(i,1:n_nonstop));
end
test_mean_length = mean(test_doc_length)
test_std_length = std(test_doc_length)

%% Part 3
n_test_not_train_ns = n_total_unique_word_ns - n_train_unique_word_ns

%% Part 4
total_word_freq_ns = zeros(n_nonstop,1);
for i = 1:n_nonstop
    total_word_freq_ns(i) = sum(s_total(:,i));
end

[sorted_total_word_freq, I] = sort(total_word_freq_ns, 'descend');
for i = 1:10
    sorted_total_word_freq(i) %frequency
    a = ia(I(i));
    words(a)     
end

%% Part 5
min_freq_word = min(total_word_freq_ns)
n_min_freq_word = 0;
for i= 1: n_nonstop
    if total_word_freq_ns(i) == min_freq_word
        n_min_freq_word = n_min_freq_word +1;
        word = words(ia(i));
        if startsWith(word,'aero')
            word
        end
    end
end

%% Train
bag_train = sparse(train_label(train(:,1)),...
    id_nonstop(train(:,2)),train(:,3));

n_word_class = zeros(20,1);
for m = 1:20
    n_word_class(m) = sum(bag_train(m,1:n_nonstop));
end
alpha = 1/n_total_unique_word_ns;
beta = (bag_train(:,1:n_nonstop) + alpha)./(n_word_class+1);
beta_log = log(beta);

py = zeros(20,1);
for i = 1:size(train_label,1)
    py(train_label(i)) = py(train_label(i)) + 1/n_train_docs;
end

%% Test
MAP = zeros(n_test_docs,1);
MAP_class = s_test(:,1:n_nonstop)*beta_log';
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

%% Part 6
CM = confusionmat(MAP, test_label);
CCR = sum(diag(CM))/n_test_docs

save('4_1e_results')