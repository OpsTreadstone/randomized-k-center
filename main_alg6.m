% Experiments of Algorithm 6.

% Arguments required:
% dataset_
%     options: ['shuttle', 'tiny_covertype', 'tiny_kddcup99', 'tiny_pokerhand']
% alg
%     options: ['alg6', 'baseline2', 'baseline3', 'baseline4', 'baseline5']
% ratio_outliers
% s_values
%     for example: [2, 4, 8, 16];
% mu_alg6
% delta_alg6
% mu_base2_values
% eps_base5_values

% To run this script, for example, use the following command:
% matlab -nodisplay -r "dataset_='shuttle';alg='alg6';ratio_outliers=0.01;s_values=[2,4,8,16];mu_alg6=0.9;delta_alg6=0.1;main_alg6;exit;"
% matlab -nodisplay -r "dataset_='shuttle';alg='baseline2';ratio_outliers=0.01;s_values=[2,4,8,16];mu_base2_values=[1,2,4];main_alg6;exit;"
% matlab -nodisplay -r "dataset_='shuttle';alg='baseline5';ratio_outliers=0.01;s_values=[2,4,8,16];eps_base5_values=[0.1,0.99];main_alg6;exit;"

repeat = 10;
if strcmp(dataset_, "cifar10") || strcmp(dataset_, "fashion_mnist") || strcmp(dataset_, "mnist") || strcmp(dataset_, "pokerhand") || strcmp(dataset_, "svhn")
    k = 10;
elseif strcmp(dataset_, "covertype") || strcmp(dataset_, "shuttle") || strcmp(dataset_, "tiny_covertype")
    k = 7;
elseif strcmp(dataset_, "kddcup99")
    k = 23;
elseif strcmp(dataset_, "tiny_kddcup99")
    k = 14;
elseif strcmp(dataset_, "tiny_pokerhand")
    k = 9;
end
alpha_baseline_7 = 4;
beta_baseline_7 = 8;
eta_baseline_7 = 16;
rho_base4_values = [2];
epsilon_for_calc_radius = 1;

disp(['Loading ', dataset_]);
data_folder = ['./datasets/', dataset_];
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
gamma = z / num_data;

save_folder_centers = strcat('./exp_res/alg6/centers/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_centers, 'dir') == 0
    mkdir(save_folder_centers)
end

save_folder_records = strcat('./exp_res/alg6/records/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_records, 'dir') == 0
    mkdir(save_folder_records)
end

if strcmp(alg, "alg6")
    for s_idx = 1:size(s_values, 2)
        s = s_values(s_idx);

        runtime_coreset = zeros(repeat, 1);
        coresets = cell(1, repeat);
        weights = cell(1, repeat);
        size_coreset = cell(1, repeat);
        commu_cost = zeros(repeat, 1);
        
        for rep = 1:repeat
            disp(['Dataset: ', dataset_, ', ratio_outliers = ',...
                num2str(ratio_outliers), ', alg = ', alg, ', s = ',...
                num2str(s), ', mu = ', num2str(mu_alg6), ', rep ',...
                num2str(rep)]);
            
            [coresets{rep}, weights{rep}, runtime_coreset(rep,:), ...
                size_coreset{rep}, commu_cost(rep,:)] = alg_6(data, ...
                k, z, s, mu_alg6, delta_alg6, epsilon_for_calc_radius);
        end
        
        disp('Calculating centers and radius...');
        alg_center_collection = ["baseline3_center"];
        for alg_center_idx = 1:size(alg_center_collection,2)
            alg_center = alg_center_collection(alg_center_idx);
            
            runtime_radius = zeros(repeat, 1);
            centers = cell(repeat, 1);
            radius_z = zeros(repeat, 1);
            radius_1_eps_z = zeros(repeat, 1);
            
            for rep = 1:repeat
                fprintf('alg_center = %s, rep %d\n', alg_center, rep);
                
                coresets_rep = zeros(num_data, dim_data);
                weights_rep = zeros(num_data, 1);
                idx_a = 1;
                for i = 1:size(coresets{1,rep}, 2)
                    idx_b = idx_a + size(coresets{1,rep}{1,i},1) - 1;
                    coresets_rep(idx_a:idx_b, :) = coresets{1,rep}{1,i};
                    weights_rep(idx_a:idx_b, :) = weights{1,rep}{1,i};
                    idx_a = idx_b + 1;
                end
                coresets_rep = coresets_rep(1:idx_b, :);
                weights_rep = weights_rep(1:idx_b, :);
                
                assert(size(coresets_rep,1) == sum(size_coreset{1,rep}));
                assert(size(weights_rep,1) == sum(size_coreset{1,rep}));
                assert(sum(weights_rep) == num_data);
                
                % calculate center
                if strcmp(alg_center, "baseline3_center")
                    [centers{rep,1}, runtime_radius(rep,:)] = ...
                        alg_baseline_3_center(data, coresets_rep, weights_rep, k, z);
                end
                
                dist_mat = pdist2(data, centers{rep,1});
                dist_mat = min(dist_mat, [], 2);
                [~, idx] = maxk(dist_mat, z+1);
                radius_z(rep, :) = dist_mat(idx(z+1));
                num_to_remove = round((1+epsilon_for_calc_radius)*z);
                [~, idx] = maxk(dist_mat, num_to_remove+1);
                radius_1_eps_z(rep, :) = dist_mat(idx(num_to_remove + 1));
            end
            
            save_filename_centers = strcat(save_folder_centers, '/k_', ...
                num2str(k), '_alg6_s_', num2str(s), '_mu_', num2str(mu_alg6), ...
                '_delta_', num2str(delta_alg6), '_', alg_center, '.mat');
            save_filename_records = strcat(save_folder_records, '/k_', ...
                num2str(k), '_alg6_s_', num2str(s), '_mu_', num2str(mu_alg6), ...
                '_delta_', num2str(delta_alg6), '_', alg_center, '.mat');
            
            save(save_filename_centers, '-v7.3', 'coresets', 'weights', 'centers');
            save(save_filename_records, '-v7.3', 'runtime_coreset', 'runtime_radius', ...
                'radius_z', 'radius_1_eps_z', 'size_coreset', 'commu_cost');
            msg = sprintf('%s %s', save_filename_records, 'saved');
            disp(msg);
            disp(" ");
        end
    end
elseif strcmp(alg, "baseline2")
    for s_idx = 1:size(s_values, 2)
        s = s_values(s_idx);

        for mu_base2_idx = 1:size(mu_base2_values, 2)
            mu_base2 = mu_base2_values(mu_base2_idx);
            
            runtime_coreset = zeros(repeat, 1);
            coresets = cell(1, repeat);
            weights = cell(1, repeat);
            size_coreset = cell(1, repeat);
            commu_cost = zeros(repeat, 1);
            precision_values = zeros(repeat, 1);
            runtime_radius = zeros(repeat, 1);
            centers = cell(repeat, 1);
            radius_z = zeros(repeat, 1);
            radius_1_eps_z = zeros(repeat, 1);
            
            for rep = 1:repeat
                disp(['Dataset: ', dataset_, ', ratio_outliers = ',...
                    num2str(ratio_outliers), ', alg = baseline2, s = ',...
                    num2str(s), ', mu_base2 = ', num2str(mu_base2), ...
                    ', rep ', num2str(rep)]);
                [coresets{rep}, weights{rep}, centers{rep,1}, radius_z(rep,:), ...
                    radius_1_eps_z(rep,:), runtime_coreset(rep,:), ...
                    runtime_radius(rep,:), size_coreset{rep}, ...
                    commu_cost(rep,:), precision_values(rep,:)] = ...
                    alg_baseline_2(data, k, z, s, mu_base2, epsilon_for_calc_radius);
            end
            
            save_filename_centers = strcat(save_folder_centers, '/k_', ...
                num2str(k), '_baseline2_s_', num2str(s), '_mu_', ...
                num2str(mu_base2), '.mat');
            save_filename_records = strcat(save_folder_records, '/k_', ...
                num2str(k), '_baseline2_s_', num2str(s), '_mu_', ...
                num2str(mu_base2), '.mat');
            
            save(save_filename_centers, '-v7.3', 'coresets', 'weights', 'centers');
            save(save_filename_records, '-v7.3', 'runtime_coreset', ...
                'runtime_radius', 'radius_z', 'radius_1_eps_z', ...
                'size_coreset', 'commu_cost', 'precision_values');
            msg = sprintf('%s %s', save_filename_records, 'saved');
            disp(msg);
            disp(" ");
        end
    end
elseif strcmp(alg, "baseline3")
    for s_idx = 1:size(s_values, 2)
        s = s_values(s_idx);

        runtime_coreset = zeros(repeat, 1);
        coresets = cell(1, repeat);
        weights = cell(1, repeat);
        size_coreset = cell(1, repeat);
        commu_cost = zeros(repeat, 1);
        runtime_radius = zeros(repeat, 1);
        centers = cell(repeat, 1);
        radius_z = zeros(repeat, 1);
        radius_1_eps_z = zeros(repeat, 1);
        
        for rep = 1:repeat
            disp(['Dataset: ', dataset_, ', ratio_outliers = ',...
                num2str(ratio_outliers), ', alg = baseline3, s = ',...
                num2str(s), ', rep ', num2str(rep)]);
            [coresets{rep}, weights{rep}, centers{rep,1}, ...
                radius_z(rep,:), radius_1_eps_z(rep,:), ...
                runtime_coreset(rep,:), runtime_radius(rep,:), ...
                size_coreset{rep}, commu_cost(rep,:)] = ...
                alg_baseline_3(data, k, z, s, epsilon_for_calc_radius);
        end
        
        save_filename_centers = strcat(save_folder_centers, '/k_', ...
            num2str(k), '_baseline3_s_', num2str(s), '.mat');
        save_filename_records = strcat(save_folder_records, '/k_', ...
            num2str(k), '_baseline3_s_', num2str(s), '.mat');
        
        save(save_filename_centers, '-v7.3', 'coresets', 'weights', 'centers');
        save(save_filename_records, '-v7.3', 'runtime_coreset', ...
            'runtime_radius', 'radius_z', 'radius_1_eps_z', ...
            'size_coreset', 'commu_cost');
        msg = sprintf('%s %s', save_filename_records, 'saved');
        disp(msg);
        disp(" ");
    end
elseif strcmp(alg, "baseline4")
    for s_idx = 1:size(s_values, 2)
        s = s_values(s_idx);

        for rho_base4_idx = 1:size(rho_base4_values, 2)
            rho_base4 = rho_base4_values(rho_base4_idx);
            
            runtime_coreset = zeros(repeat, 1);
            coresets = cell(1, repeat);
            weights = cell(1, repeat);
            size_coreset = cell(1, repeat);
            commu_cost = zeros(repeat, 1);
            runtime_radius = zeros(repeat, 1);
            centers = cell(repeat, 1);
            radius_z = zeros(repeat, 1);
            radius_1_eps_z = zeros(repeat, 1);
            
            for rep = 1:repeat
                disp(['Dataset: ', dataset_, ', ratio_outliers = ',...
                    num2str(ratio_outliers), ', alg = baseline5, s = ',...
                    num2str(s), ', rho_base4 = ', num2str(rho_base4), ...
                    ', rep ', num2str(rep)]);
                [coresets{rep}, weights{rep}, centers{rep,1}, ...
                    radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime_coreset(rep,:), runtime_radius(rep,:), ...
                    size_coreset{rep}, commu_cost(rep,:)] = ...
                    alg_baseline_4(data, k, z, s, rho_base4, epsilon_for_calc_radius);
            end
            
            save_filename_centers = strcat(save_folder_centers, '/k_', ...
                num2str(k), '_baseline4_s_', num2str(s), '_rho_', ...
                num2str(rho_base4), '.mat');
            save_filename_records = strcat(save_folder_records, '/k_', ...
                num2str(k), '_baseline4_s_', num2str(s), '_rho_', ...
                num2str(rho_base4), '.mat');
            
            save(save_filename_centers, '-v7.3', 'coresets', 'weights', 'centers');
            save(save_filename_records, '-v7.3', 'runtime_coreset', ...
                'runtime_radius', 'radius_z', 'radius_1_eps_z', ...
                'size_coreset', 'commu_cost');
            msg = sprintf('%s %s', save_filename_records, 'saved');
            disp(msg);
            disp(" ");
        end
    end
elseif strcmp(alg, "baseline5")
    for s_idx = 1:size(s_values, 2)
        s = s_values(s_idx);

        for eps_base5_idx = 1:size(eps_base5_values, 2)
            eps_base5 = eps_base5_values(eps_base5_idx);
            
            runtime_coreset = zeros(repeat, 1);
            coresets = cell(1, repeat);
            weights = cell(1, repeat);
            size_coreset = cell(1, repeat);
            commu_cost = zeros(repeat, 1);
            runtime_radius = zeros(repeat, 1);
            centers = cell(repeat, 1);
            radius_z = zeros(repeat, 1);
            radius_1_eps_z = zeros(repeat, 1);
            
            for rep = 1:repeat
                disp(['Dataset: ', dataset_, ', ratio_outliers = ',...
                    num2str(ratio_outliers), ', alg = baseline5, s = ',...
                    num2str(s), ', eps_base5 = ', num2str(eps_base5), ...
                    ', rep ', num2str(rep)]);
                [coresets{rep}, weights{rep}, centers{rep,1}, ...
                    radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime_coreset(rep,:), runtime_radius(rep,:), ...
                    size_coreset{rep}, commu_cost(rep,:)] = ...
                    alg_baseline_5(data, k, z, s, eps_base5, epsilon_for_calc_radius);
            end
            
            save_filename_centers = strcat(save_folder_centers, '/k_', ...
                num2str(k), '_baseline5_s_', num2str(s), '_eps_', ...
                num2str(eps_base5), '.mat');
            save_filename_records = strcat(save_folder_records, '/k_', ...
                num2str(k), '_baseline5_s_', num2str(s), '_eps_', ...
                num2str(eps_base5), '.mat');
            
            save(save_filename_centers, '-v7.3', 'coresets', 'weights', 'centers');
            save(save_filename_records, '-v7.3', 'runtime_coreset', ...
                'runtime_radius', 'radius_z', 'radius_1_eps_z', ...
                'size_coreset', 'commu_cost');
            msg = sprintf('%s %s', save_filename_records, 'saved');
            disp(msg);
            disp(" ");
        end
    end
end
