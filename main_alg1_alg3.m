% Experiments of Algorithm 1 and Algorithm 3.
% Force Alg 1 and Alg 3 to output the same number of points

% Arguments required:
% dataset_
%     options: ['shuttle', 'tiny_covertype', 'tiny_kddcup99', 'tiny_pokerhand']
% alg
%     options: ['alg1', 'alg3', 'baseline1']
% ratio_outliers
% k_min
% k_max

% To run this script, for example, use the following command:
% matlab -nodisplay -r "dataset_='shuttle';alg='alg1';ratio_outliers=0.01;k_min=4;k_max=6;main_alg1_alg3;exit;"

disp(['dataset_: ', dataset_]);
disp(['alg: ', alg, ', ratio_outliers: ', num2str(ratio_outliers)]);
disp(" ");
pause(5);

repeat = 10;
eta = 0.1;

data_folder = ['./datasets/', dataset_];
save_file_t_values_alg1 = strcat(data_folder, '/', dataset_, '_t_values_alg1.mat');
save_file_num_centers_alg3 = strcat(data_folder, '/', dataset_, '_num_centers_alg3.mat');
load(save_file_t_values_alg1);
load(save_file_num_centers_alg3);

disp(['Loading ', dataset_]);
load([data_folder, '/', dataset_, '.mat']);
outlier_file = [data_folder, '/', dataset_, '_outliers_', num2str(ratio_outliers), '.mat'];
load(outlier_file);
ori_num_data = size(data, 1);
data = [data; generated_outliers];
[num_data, dim_data] = size(data);
rand_idx = randperm(num_data);
data = data(rand_idx, :);
rand_idx = randperm(num_data);
data = data(rand_idx, :);
rand_idx = randperm(num_data);
data = data(rand_idx, :);
disp(['Finish loading ', dataset_]);
disp(" ");

z = size(generated_outliers, 1);

save_folder_centers = strcat('./exp_res/alg1_alg3/centers/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_centers, 'dir') == 0
    mkdir(save_folder_centers)
end

save_folder_records = strcat('./exp_res/alg1_alg3/records/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_records, 'dir') == 0
    mkdir(save_folder_records)
end

for k = k_min:2:k_max
    for epsilon = 0.2:0.2:1
        idx_1 = round(ratio_outliers * 100);
        idx_2 = round(k / 2 - 1);
        idx_3 = round(epsilon * 10 / 2);
        t = t_values_alg1(idx_1, idx_2, idx_3);
        target_num = num_centers_alg3(idx_1, idx_2, idx_3);
        
        save_filename_centers = strcat(save_folder_centers, '/k_', ...
            num2str(k), '_', alg, '_epsilon_', num2str(epsilon), '.mat');
        save_filename_records = strcat(save_folder_records, '/k_', ...
            num2str(k), '_', alg, '_epsilon_', num2str(epsilon), '.mat');
        
        centers = [];
        radius_z = zeros(repeat, 1);
        radius_1_eps_z = zeros(repeat, 1);
        runtime = zeros(repeat, 1);
        num_centers = zeros(repeat, 1);
        
        for rep = 1:repeat
            disp(['Dataset: ', dataset_, ', ratio_outliers = ', ...
                num2str(ratio_outliers), ', alg = ', alg, ', k = ', ...
                num2str(k), ', epsilon = ', num2str(epsilon), ...
                ', rep ', num2str(rep)]);

            if strcmp(alg, "alg1")
                [centers_rep, radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime(rep,:)] = alg_1_for_comp_with_alg3(data, z, ...
                    epsilon, eta, t, target_num);
            elseif strcmp(alg, "alg3")
                [centers_rep, radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime(rep,:)] = alg_3(data, k, z, epsilon, eta);
            elseif strcmp(alg, "baseline1")
                [centers_rep, radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime(rep,:)] = alg_baseline_1(data, target_num, z, epsilon);
            end
            centers = cat(3, centers, centers_rep);
            num_centers(rep, :) = size(centers_rep, 1);
            
            assert(isequal(size(centers_rep), [target_num,dim_data]));
            assert(sum(sum(centers_rep==zeros(target_num,dim_data),2)==dim_data) == 0);
        end
        
        assert(isequal(size(radius_z), [repeat,1]));
        assert(isequal(size(radius_1_eps_z), [repeat,1]));
        assert(isequal(size(runtime), [repeat,1]));
        assert(isequal(size(num_centers), [repeat,1]));
        
        assert(sum(radius_z==zeros(repeat,1)) == 0);
        assert(sum(radius_1_eps_z==zeros(repeat,1)) == 0);
        assert(sum(runtime==zeros(repeat,1)) == 0);
        assert(sum(num_centers==zeros(repeat,1)) == 0);
        
        save(save_filename_centers, '-v7.3', 'centers');
        save(save_filename_records, '-v7.3', 'radius_z', ...
            'radius_1_eps_z', 'runtime', 'num_centers');
        msg = sprintf('%s %s', save_filename_records, 'saved');
        disp(msg);
        disp(" ");
    end
end
