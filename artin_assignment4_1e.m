clc
clear
load 4_1d_results.mat

%% Train & Test
CCR = zeros(14,1);
for k = -5:0.5:1.5
    alpha = 10^k;
    beta = (bag_train + alpha)./(n_word_class+n_total_unique_word*alpha);
    beta_log = log(beta);

    MAP = zeros(n_test_docs,1); %maximum aposteriori probability
    MAP_class = s_test*beta_log';
    for j = 1:n_test_docs
        AP_log = MAP_class(j,:); %log(aposteriori probability of each class)
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
    CM = confusionmat(MAP, test_label);
    ind = (k+5)/0.5 + 1;
    CCR(ind) = sum(diag(CM))/n_test_docs;
end
%% Plot
semilogx(10.^[-5:0.5:1.5],CCR,'-o');
title('Test Errors for Different Choices of \alpha');
grid on
xlabel('\alpha -1');
ylabel('CCR');
savefig('4_1e_CCR.fig');
saveas(gcf,'4_1e_CCR.jpg');

save('4_1e_results')