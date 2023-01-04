% Experiments of Algorithm 5.

% Arguments required:
% dataset_
%     options: ['shuttle', 'tiny_covertype', 'tiny_kddcup99', 'tiny_pokerhand']
% alg_coreset
%     the coreset construction algorithm
%     options: ['alg5_using_alg1', 'baseline2', 'uniform', 'none']
%     'none' means treating the whole dataset as the coreset
% ratio_outliers
% ratio_alg5
%     the ratio of the size of coreset to that of the whole dataset
% eta = 0.1

% To run this script, for example, use the following command:
% matlab -nodisplay -r "dataset_='shuttle';alg_coreset='alg5_using_alg1';ratio_outliers=0.01;ratio_alg5=0.03;eta=0.1;main_alg5;exit;"

if strcmp(alg_coreset, "none")
    alg_center_collection = ["baseline6", "baseline7"];
else
    alg_center_collection = ["baseline3_center"];
end

disp(['dataset_: ', dataset_]);
disp(['alg_coreset: ', alg_coreset]);
disp(['ratio_outliers: ', num2str(ratio_outliers)]);
disp(" ");
pause(5);

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
epsilon = 1;
alpha_baseline_7 = 4;
beta_baseline_7 = 8;
eta_baseline_7 = 16;

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

save_folder_centers = strcat('./exp_res/alg5/centers/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_centers, 'dir') == 0
    mkdir(save_folder_centers)
end

save_folder_records = strcat('./exp_res/alg5/records/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_records, 'dir') == 0
    mkdir(save_folder_records)
end

runtime_coreset = zeros(repeat, 1);
runtime_radius = zeros(repeat, 1);
coresets = cell(repeat, 1);
weights = cell(repeat, 1);
centers = cell(repeat, 1);
radius_z = zeros(repeat, 1);
radius_1_eps_z = zeros(repeat, 1);
size_coreset = zeros(repeat, 1);
precision_base2 = zeros(repeat, 1);

disp('Calculating coreset...');
if strcmp(alg_coreset, "alg5_using_alg1")
    for rep = 1:repeat
        disp(['alg_coreset = alg5_using_alg1, ratio_alg5 = ', ...
            num2str(ratio_alg5), ', rep ', num2str(rep)]);
        
        [coresets{rep,1}, weights{rep,1}, runtime_coreset(rep,:), precision_base2(rep,:)] = ...
            alg_5_using_alg_1_deterministic(data, k, z, eta, ratio_alg5);
        size_coreset(rep, :) = size(coresets{rep,1}, 1);
        
        [num_coresets, dim_coresets] = size(coresets{rep,1});
        [num_weights, dim_weights] = size(weights{rep,1});
        assert(sum(sum(coresets{rep,1}==zeros(num_coresets,dim_coresets),2)==dim_coresets) == 0);
        assert(sum(weights{rep,1}) == num_data);
    end
elseif strcmp(alg_coreset, "baseline2")
    size_coreset_file = strcat(save_folder_records, '/k_', num2str(k), ...
        '_alg5_using_alg1_eta_0.1_ratio_', num2str(ratio_alg5), '_baseline3_center.mat');
    size_coreset_alg5 = load(size_coreset_file, 'size_coreset');
    
    precision_values = zeros(repeat, 1);
    for rep = 1:repeat
        disp(['alg_coreset = baseline2_ratio_', num2str(ratio_alg5), ...
            ', rep ', num2str(rep)]);
        [coresets{rep,1}, weights{rep,1}, centers{rep,1}, ~, ~, ...
            runtime_coreset(rep,:), runtime_radius(rep,:), ...
            size_coreset(rep,:), precision_values(rep,:)] = ...
            alg_baseline_2_for_comp_with_alg5(data, k, z, 1, ...
                size_coreset_alg5.size_coreset(rep), epsilon);
    end
    assert(isequal(size_coreset, size_coreset_alg5.size_coreset));
elseif strcmp(alg_coreset, "uniform")
    records_file = strcat(save_folder_records, '/k_', num2str(k), ...
        '_alg5_using_alg1_eta_0.1_ratio_', num2str(ratio_alg5), '_baseline3_center.mat');
    alg5_records = load(records_file);
    
    for rep = 1:repeat
        disp(['alg_coreset = uniform_ratio_', num2str(ratio_alg5), ...
            ', rep ', num2str(rep)]);
        
        [coresets{rep,1}, weights{rep,1}, runtime_coreset(rep,:)] = ...
            alg_uniform(data, alg5_records.size_coreset(rep));
        size_coreset(rep, :) = size(coresets{rep,1}, 1);
        
        [num_coresets, dim_coresets] = size(coresets{rep,1});
        [num_weights, dim_weights] = size(weights{rep,1});
        assert(sum(sum(coresets{rep,1}==zeros(num_coresets,dim_coresets),2)==dim_coresets) == 0);
    end
    assert(isequal(size_coreset, alg5_records.size_coreset));
elseif strcmp(alg_coreset, "none")
    for rep = 1:repeat
        coresets{rep, 1} = data;
        weights{rep, 1} = ones(size(data,1), 1);
        runtime_coreset(rep, :) = 0;
        size_coreset(rep, :) = size(data, 1);
    end
end

disp('Calculating centers and radius...');
if strcmp(alg_coreset, "baseline2")
    for rep = 1:repeat
        disp(['alg_center = baseline2, rep ', num2str(rep)]);
        
        dist_mat = pdist2(data, centers{rep,1});
        dist_mat = min(dist_mat, [], 2);
        [~, idx] = maxk(dist_mat, z+1);
        radius_z(rep, :) = dist_mat(idx(z+1));
        num_to_remove = round((1+epsilon)*z);
        [~, idx] = maxk(dist_mat, num_to_remove+1);
        radius_1_eps_z(rep, :) = dist_mat(idx(num_to_remove + 1));
    end
    
    save_filename_centers = strcat(save_folder_centers, '/k_', ...
        num2str(k), '_', alg_coreset, '_ratio_', num2str(ratio_alg5), ...
        '_baseline2.mat');
    save_filename_records = strcat(save_folder_records, '/k_', ...
        num2str(k), '_', alg_coreset, '_ratio_', num2str(ratio_alg5), ...
        '_baseline2.mat');

    save(save_filename_centers, '-v7.3', 'coresets', 'weights', 'centers');
    save(save_filename_records, '-v7.3', 'runtime_coreset', ...
        'runtime_radius', 'radius_z', 'radius_1_eps_z', 'size_coreset', ...
        'precision_values');
    msg = sprintf('%s %s', save_filename_records, 'saved');
    disp(msg);
    disp(" ");
else
    for alg_center_idx = 1:size(alg_center_collection,2)
        alg_center = alg_center_collection(alg_center_idx);

        runtime_radius = zeros(repeat, 1);
        radius_z = zeros(repeat, 1);
        radius_1_eps_z = zeros(repeat, 1);

        for rep = 1:repeat
            fprintf('alg_center = %s, rep %d\n', alg_center, rep);

            % calculate center
            if strcmp(alg_center, "baseline6")
                [centers{rep,1}, runtime_radius(rep,:)] = ...
                    alg_baseline_6_weighted(coresets{rep,1}, weights{rep,1}, k, z);
            elseif strcmp(alg_center, "baseline7")
                [centers{rep,1}, runtime_radius(rep,:)] = ...
                    alg_baseline_7_weighted(coresets{rep,1}, weights{rep,1}, k, z, ...
                    alpha_baseline_7, beta_baseline_7, eta_baseline_7);
            elseif strcmp(alg_center, "baseline3_center")
                [centers{rep,1}, runtime_radius(rep,:)] = ...
                    alg_baseline_3_center(data, coresets{rep,1}, weights{rep,1}, k, z);
            end

            dist_mat = pdist2(data, centers{rep,1});
            dist_mat = min(dist_mat, [], 2);
            [~, idx] = maxk(dist_mat, z+1);
            radius_z(rep, :) = dist_mat(idx(z+1));
            num_to_remove = round((1+epsilon)*z);
            [~, idx] = maxk(dist_mat, num_to_remove+1);
            radius_1_eps_z(rep, :) = dist_mat(idx(num_to_remove + 1));
        end
        
        if strcmp(alg_coreset, "none")
            save_filename_centers = strcat(save_folder_centers, '/k_', ...
                num2str(k), '_none_', alg_center, '.mat');
            save_filename_records = strcat(save_folder_records, '/k_', ...
                num2str(k), '_none_', alg_center, '.mat');
        elseif strcmp(alg_coreset, "alg5_using_alg1")
            save_filename_centers = strcat(save_folder_centers, '/k_', ...
                num2str(k), '_', alg_coreset, '_eta_', num2str(eta), ...
                '_ratio_', num2str(ratio_alg5), '_', alg_center, '.mat');
            save_filename_records = strcat(save_folder_records, '/k_', ...
                num2str(k), '_', alg_coreset, '_eta_', num2str(eta), ...
                '_ratio_', num2str(ratio_alg5), '_', alg_center, '.mat');
        else
            save_filename_centers = strcat(save_folder_centers, '/k_', ...
                num2str(k), '_', alg_coreset, '_ratio_', num2str(ratio_alg5), ...
                '_', alg_center, '.mat');
            save_filename_records = strcat(save_folder_records, '/k_', ...
                num2str(k), '_', alg_coreset, '_ratio_', num2str(ratio_alg5), ...
                '_', alg_center, '.mat');
        end

        save(save_filename_centers, '-v7.3', 'coresets', 'weights', 'centers');
        if strcmp(alg_coreset, "alg5_using_alg1")
            save(save_filename_records, '-v7.3', 'runtime_coreset', 'runtime_radius', ...
                'radius_z', 'radius_1_eps_z', 'size_coreset', 'precision_base2');
        else
            save(save_filename_records, '-v7.3', 'runtime_coreset', 'runtime_radius', ...
                'radius_z', 'radius_1_eps_z', 'size_coreset');
        end
        msg = sprintf('%s %s', save_filename_records, 'saved');
        disp(msg);
        disp(" ");
    end
end
