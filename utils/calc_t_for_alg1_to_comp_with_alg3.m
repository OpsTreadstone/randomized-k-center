% Calculate the value of t for Algorithm 1 to output the same number of
% points as Algorithm 3

ratio_outliers_values = [0.01];
k_values = 4:2:20;
epsilon_values = 0.2:0.2:1;
eta = 0.1;

dataset_ = 'shuttle';
data_folder = ['../datasets/', dataset_];
load([data_folder, '/', dataset_, '.mat']);
num_data = size(data, 1);

save_folder = strcat('../datasets/', dataset_);
if exist(save_folder, 'dir') == 0
    mkdir(save_folder)
end
save_file_t_values_alg1 = strcat(data_folder, '/', dataset_, '_t_values_alg1.mat');
save_file_num_centers_alg3 = strcat(data_folder, '/', dataset_, '_num_centers_alg3.mat');

t_values_alg1 = zeros(size(ratio_outliers_values,2), size(k_values, 2), size(epsilon_values, 2));
num_centers_alg3 = zeros(size(ratio_outliers_values,2), size(k_values, 2), size(epsilon_values, 2));

for ro_idx = 1:size(ratio_outliers_values, 2)
    ratio_outliers = ratio_outliers_values(ro_idx);
    z = round(num_data * ratio_outliers);
    
    gamma = z / (num_data+z);
    
    for k_idx = 1:size(k_values, 2)
        k = k_values(k_idx);
        for eps_idx = 1:size(epsilon_values, 2)
            epsilon = epsilon_values(eps_idx);
            sigma = 2 / (1 + sqrt(1 + 4*(1+epsilon)/3/epsilon));
            n_prime = round(3 / sigma / sigma / (1+epsilon) / gamma * log(4/eta));
            t_alg3 = round((k+sqrt(k)) / (1-eta));
            num_centers_init_alg3 = round(1 / (1-gamma) * log(1/eta));
            num_centers_iter_alg3 = round((1+sigma) * (1+epsilon) * gamma * n_prime);
            num_centers_total_alg3 = num_centers_init_alg3 + (t_alg3-1) * num_centers_iter_alg3;
            
            t_alg1 = round((k+sqrt(k)) / (1-eta));
            num_centers_init_alg1 = round(1 / (1-gamma) * log(1/eta));
            num_centers_iter_alg1 = round((1+epsilon) / epsilon * log(1/eta));
            num_centers_total_alg1 = num_centers_init_alg1 + (t_alg1-1) * num_centers_iter_alg1;
            while (num_centers_total_alg1 < num_centers_total_alg3)
                t_alg1 = t_alg1 + 1;
                num_centers_total_alg1 = num_centers_init_alg1 + (t_alg1-1) * num_centers_iter_alg1;
            end
            if (num_centers_total_alg1 > num_centers_total_alg3)
                t_alg1 = t_alg1 - 1;
                num_centers_total_alg1 = num_centers_init_alg1 + (t_alg1-1) * num_centers_iter_alg1;
            end
            
            assert(num_centers_total_alg1 <= num_centers_total_alg3);
            disp(['outliers=', num2str(ratio_outliers), ', k=', num2str(k),...
                ', eps=', num2str(epsilon), ', t_alg1=', num2str(t_alg1),...
                ', alg1=', num2str(num_centers_total_alg1),...
                ', alg3=', num2str(num_centers_total_alg3)]);
            
            idx_1 = round(ratio_outliers * 100);
            idx_2 = round(k / 2 - 1);
            idx_3 = round(epsilon * 10 / 2);
            t_values_alg1(idx_1, idx_2, idx_3) = t_alg1;
            num_centers_alg3(idx_1, idx_2, idx_3) = num_centers_total_alg3;
        end
    end
end

save(save_file_t_values_alg1, 't_values_alg1');
save(save_file_num_centers_alg3, 'num_centers_alg3');
