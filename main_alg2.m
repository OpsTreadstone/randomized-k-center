% Experiments of Algorithm 2.

% Arguments required:
% dataset_
%     options: ['shuttle', 'tiny_covertype', 'tiny_kddcup99', 'tiny_pokerhand']
% alg
%     options: ['alg2', 'baseline1', 'baseline6', 'baseline7']
% ratio_outliers
% k_min
% k_max

% To run this script, for example, use the following command:
% matlab -nodisplay -r "dataset_='shuttle';alg='alg2';ratio_outliers=0.01;k_min=2;k_max=3;main_alg2;exit;"

disp(['dataset_: ', dataset_]);
disp(['alg: ', alg, ', ratio_outliers: ', num2str(ratio_outliers)]);
disp(" ");
pause(5);

eta = 0.1;
epsilon = 1;
repeat = 10;
alpha_baseline_7 = 4;
beta_baseline_7 = 8;
eta_baseline_7 = 16;

data_folder = ['./datasets/', dataset_];
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

z = ori_num_data * ratio_outliers;
gamma = z / num_data;

save_folder_centers = strcat('./exp_res/alg2/centers/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_centers, 'dir') == 0
    mkdir(save_folder_centers)
end

save_folder_records = strcat('./exp_res/alg2/records/', dataset_, '/ratio_outliers_', ...
    num2str(ratio_outliers));
if exist(save_folder_records, 'dir') == 0
    mkdir(save_folder_records)
end

for k = k_min:1:k_max
    save_filename_centers = strcat(save_folder_centers, '/k_', ...
        num2str(k), '_', alg, '_epsilon_', num2str(epsilon), '.mat');
    save_filename_records = strcat(save_folder_records, '/k_', ...
        num2str(k), '_', alg, '_epsilon_', num2str(epsilon), '.mat');
    
    if strcmp(alg, "alg2")
        num_iterations = round(log(10) * ((1+epsilon)/epsilon)^(k-1) / (1-gamma));
    end
    
    if strcmp(alg, "alg2")
        centers_z = [];
        centers_1_eps_z = [];
    else
        centers = [];
    end
    radius_z = zeros(repeat, 1);
    radius_1_eps_z = zeros(repeat, 1);
    runtime = zeros(repeat, 1);
    
    for rep = 1:10
        disp(['Dataset: ', dataset_, ', ratio_outliers = ', ...
            num2str(ratio_outliers), ', alg = ', alg, ', k = ', ...
            num2str(k), ', rep = ', num2str(rep)]);
        radius_z(rep, :) = Inf;
        radius_1_eps_z(rep, :) = Inf;
        if strcmp(alg, "alg2")
            for iter = 1:num_iterations
                disp(['Dataset: ', dataset_, ', ratio_outliers = ', ...
                    num2str(ratio_outliers), ', alg = ', alg, ...
                    ', iter = ', num2str(iter), ...
                    ', num_iterations = ', num2str(num_iterations)]);
                
                tic
                [centers_iter, radius_z_iter, radius_1_eps_z_iter] = ...
                    alg_2(data, k, z, epsilon);
                runtime(rep, :) = runtime(rep,:) + toc;

                if radius_z_iter < radius_z(rep, :)
                    radius_z(rep, :) = radius_z_iter;
                    centers_z_rep = centers_iter;
                end
                if radius_1_eps_z_iter < radius_1_eps_z(rep, :)
                    radius_1_eps_z(rep, :) = radius_1_eps_z_iter;
                    centers_1_eps_z_rep = centers_iter;
                end

                assert(isequal(size(centers_iter), [k,dim_data]));
                assert(sum(sum(centers_iter==zeros(k,dim_data),2)==dim_data) == 0);
            end
            centers_z = cat(3, centers_z, centers_z_rep);
            centers_1_eps_z = cat(3, centers_1_eps_z, centers_1_eps_z_rep);
        elseif strcmp(alg, "baseline1")
            [centers_rep, radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime(rep,:)] = alg_baseline_1(data, k, z, epsilon);
            centers = cat(3, centers, centers_rep);
        elseif strcmp(alg, "baseline6")
            [centers_rep, radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime(rep,:)] = alg_baseline_6(data, k, z, epsilon);
            centers = cat(3, centers, centers_rep);
        elseif strcmp(alg, "baseline7")
            [centers_rep, radius_z(rep,:), radius_1_eps_z(rep,:), ...
                    runtime(rep,:)] = alg_baseline_7(data, k, z, ...
                    alpha_baseline_7, beta_baseline_7, eta_baseline_7, epsilon);
            centers = cat(3, centers, centers_rep);
        end
    end
    
    assert(isequal(size(radius_z), [repeat,1]));
    assert(isequal(size(radius_1_eps_z), [repeat,1]));
    assert(isequal(size(runtime), [repeat,1]));
    
    assert(sum(radius_z==zeros(repeat,1)) == 0);
    assert(sum(radius_1_eps_z==zeros(repeat,1)) == 0);
    assert(sum(runtime==zeros(repeat,1)) == 0);
    
    if strcmp(alg, "alg2")
        save(save_filename_centers, '-v7.3', 'centers_z', 'centers_1_eps_z');
    else
        save(save_filename_centers, '-v7.3', 'centers');
    end
    save(save_filename_records, '-v7.3', 'radius_z', 'radius_1_eps_z', ...
        'runtime');
    msg = sprintf('%s %s', save_filename_records, 'saved');
    disp(msg);

    disp(" ");
end
