% The implementation of the "MKC+" algorithm of
% Fast Distributed k-Center Clustering with Outliers on Massive Data

% m: number of machines
% G: a guess of the optimal solution's value
function [coresets_ret, weights_ret, centers, radius_z, radius_1_eps_z, ...
    runtime_coreset, runtime_radius, size_coreset, commu_cost] = ...
    alg_baseline_3(data, k, z, m, epsilon_for_calc_radius)
    tic
    
    [num_data, dim_data] = size(data);
    num_data_machine = floor(num_data / m);
    coresets_ret = cell(1, m);
    weights_ret = cell(1, m);
    size_coreset = ones(m, 1) * (k+z);
    commu_cost = m*(k+z)*dim_data + m*(k+z);
    coresets = zeros(m*(k+z), dim_data);
    weights = zeros(m*(k+z), 1);
    
    for i = 1:m
        data_idx_left = (i-1) * num_data_machine + 1;
        if i == m
            data_idx_right = num_data;
        else
            data_idx_right = i * num_data_machine;
        end
        curr_data = data(data_idx_left:data_idx_right, :);
        num_curr_data = size(curr_data, 1);
        
        centers_i = alg_gonzalez_for_base3(curr_data, k+z);
        assert(size_coreset(i) == size(centers_i,1));
        coresets_ret{i} = centers_i;
        coresets((i-1)*(k+z)+1:i*(k+z), :) = centers_i;
        
        dist_mat = pdist2(curr_data, centers_i);
        [~, idx] = min(dist_mat, [], 2);
        
        weight_i = zeros(k+z, 1);
        for j = 1:num_curr_data
            weight_i(idx(j)) = weight_i(idx(j)) + 1;
        end
        assert(sum(weight_i) == num_curr_data);
        weights_ret{i} = weight_i;
        weights((i-1)*(k+z)+1:i*(k+z), :) = weight_i;
    end
    runtime_coreset = toc;
    
    tic
    rand_idx = randi(num_data);
    dist_mat = pdist2(data, data(rand_idx,:));
    lower_bound = 0;
    upper_bound = max(dist_mat);
    
    success = 0;
    feasible_centers = [];
    while upper_bound-lower_bound >= 0.01
        G = (lower_bound + upper_bound) / 2;
        [centers, weight_U_prime] = alg_cluster(coresets, weights, k, G);
        if weight_U_prime <= z
            success = 1;
            feasible_centers = centers;
            upper_bound = G;
        else
            lower_bound = G;
        end
    end
    assert(success && ~isempty(feasible_centers));
    centers = feasible_centers;
    runtime_radius = toc;    
    
    dist_mat = pdist2(data, centers);
    dist_mat = min(dist_mat, [], 2);
    [~, idx] = maxk(dist_mat, z+1);
    radius_z = dist_mat(idx(z+1));
    num_to_remove = round((1+epsilon_for_calc_radius)*z);
    [~, idx] = maxk(dist_mat, num_to_remove+1);
    radius_1_eps_z = dist_mat(idx(num_to_remove + 1));
end

function centers = alg_gonzalez_for_base3(data, k)
    [num_data, dim_data] = size(data);
    centers = zeros(k, dim_data);
    
    init_center_idx = randi(num_data);
    centers(1, :) = data(init_center_idx, :);
    dist_mat = pdist2(data, centers(1,:));
    
    for i = 2:k
        [~, nxt_center_idx] = max(dist_mat);
        centers(i, :) = data(nxt_center_idx, :);
        
        if i == k
            break
        end
        
        tmp_dist_mat = pdist2(data, centers(i,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
    end
end

function [centers, weight_U_prime] = alg_cluster(data, weight, k, G)
    [num_data, dim_data] = size(data);
    dist_mat = pdist(data);
    dist_mat = squareform(dist_mat);
    
    num_centers = 0;
    centers = zeros(k, dim_data);
    weight_redefined = zeros(num_data, 1);
    is_removed = logical(zeros(num_data, 1));
    
    while num_centers<k && sum(is_removed)~=num_data
        for i = 1:num_data
            condition = dist_mat(i,:)<=5*G;
            weight_redefined(i, :) = sum(weight(condition, :));
        end
        
        [~, idx_max_weight] = max(weight_redefined);
        num_centers = num_centers + 1;
        centers(num_centers, :) = data(idx_max_weight, :);
        
        condition = dist_mat(idx_max_weight,:)<=11*G;
        is_removed(condition, :) = 1;
        weight(condition, :) = 0;
        assert(sum(weight(is_removed,:)) == 0);
    end
    if sum(is_removed)~=num_data
        weight_U_prime = sum(weight(~is_removed));
    else
        weight_U_prime = 0;
    end
    
    if num_centers < k
        for i = 1:num_data
            condition = dist_mat(i,:)<=5*G;
            weight_redefined(i, :) = sum(weight(condition, :));
        end
        [~, idx] = maxk(weight_redefined, k-num_centers);
        centers(num_centers+1:k, :) = data(idx, :);
    end
end
