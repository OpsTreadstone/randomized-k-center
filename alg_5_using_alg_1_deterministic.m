% The implementation of Algorithm 5 that output a coreset of fixed size.

function [coreset, weight, runtime, precision_base2] = alg_5_using_alg_1_deterministic(data, k, z, eta, ratio)
    tic
    
    [num_data, dim_data] = size(data);
    epsilon = 1;
    
    num_to_exclude = 2 * z;
    
    [coreset_alg_1, others, precision_base2] = alg_1_for_alg_5_deterministic(data, k, z, epsilon, eta, ratio);
    size_coreset_alg_1 = size(coreset_alg_1, 1);
    if precision_base2 > 1.0
        precision_base2 = 1.0;
    end
    precision_base2 = precision_base2 / 6;
    
    assert(size_coreset_alg_1+size(others,1) == size(data,1));
    
    if num_data <= 100000
        dist_mat = pdist2(others, coreset_alg_1);
        [dist_mat, idx] = min(dist_mat, [], 2);
    else
        dist_mat = zeros(size(others,1), 1);
        idx = zeros(size(others,1), 1);
        others_left_idx = 1;
        while others_left_idx < size(others,1)
            others_right_idx = min([others_left_idx+50000, size(others,1)]);
            tmp_dist_mat = pdist2(others(others_left_idx:others_right_idx,:), coreset_alg_1);
            [dist_mat(others_left_idx:others_right_idx,:), idx(others_left_idx:others_right_idx,:)] = min(tmp_dist_mat, [], 2);
            others_left_idx = others_right_idx + 1;
        end
    end
    [~, outliers_idx] = maxk(dist_mat, num_to_exclude);
    
    size_coreset_total = size_coreset_alg_1 + num_to_exclude;
    coreset = zeros(size_coreset_total, dim_data);
    coreset(1:size_coreset_alg_1, :) = coreset_alg_1;
    coreset(size_coreset_alg_1+1:size_coreset_total, :) = others(outliers_idx, :);
    
    num_others = size(others, 1);
    is_outlier = logical(zeros(num_others, 1));
    is_outlier(outliers_idx, :) = 1;
    
    weight = ones(size_coreset_total, 1);
    for i = 1:num_others
        if ~is_outlier(i,:)
            weight(idx(i)) = weight(idx(i)) + 1; 
        end
    end
    assert(sum(weight) == num_data);
    
    runtime = toc;
end

function [centers, non_centers, precision] = alg_1_for_alg_5_deterministic(data, k, z, epsilon, eta, ratio)
    [num_data, dim_data] = size(data);
    gamma = z / num_data;
    
    num_centers_init = round(1 / (1-gamma) * log(1/eta));
    num_points_iter = round((1+epsilon) * z);
    num_centers_iter = round((1+epsilon) / epsilon * log(1/eta));
    
    num_output = round(num_data*ratio);
    assert(num_output > 2*z+num_centers_init);
    num_output = num_output - 2*z - num_centers_init;
    num_iteration = ceil(num_output / num_centers_iter);
    num_centers = num_centers_init + num_iteration*num_centers_iter;
    
    centers = zeros(num_centers, dim_data);
    
    is_center = logical(zeros(num_data, 1));
    
    init_centers_idx = randperm(num_data, num_centers_init);
    centers(1:num_centers_init, :) = data(init_centers_idx, :);
    is_center(init_centers_idx, :) = 1;
    
    num_centers_sofar = num_centers_init;
    dist_mat = pdist2(data, centers(1:num_centers_init,:));
    dist_mat = min(dist_mat, [], 2);
    
    radius_kz_ready = 0;
    if num_centers_sofar >= k+z
        radius_kz_ready = 1;
        radius_kz = max(dist_mat);
    end
    
    for j = 1:num_iteration
        [~, Qj_idx] = maxk(dist_mat, num_points_iter);
        idx_in_Qj = randperm(num_points_iter, num_centers_iter);
        centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter, :) = data(Qj_idx(idx_in_Qj), :);
        
        is_center(Qj_idx(idx_in_Qj), :) = 1;
        
        tmp_dist_mat = pdist2(data, centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
        
        num_centers_sofar = num_centers_sofar + num_centers_iter;
        
        if (~radius_kz_ready) && (num_centers_sofar>=k+z)
            radius_kz_ready = 1;
            radius_kz = max(dist_mat);
        end
    end
    assert(num_centers_sofar == num_centers);
    
    non_centers = data(~is_center, :);
    
    radius = max(dist_mat);
    precision = radius / radius_kz * 2;
end
