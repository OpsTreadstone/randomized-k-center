% The implementation of the "CPP" algorithm of
% Solving k-center Clustering (with Outliers) in MapReduce and Streaming,
% almost as Accurately as Sequentially

% This version requires a parameter mu and each machine output mu*(k+z)
% points

function [coresets_ret, weights_ret, centers, radius_z, radius_1_eps_z, ...
    runtime_coreset, runtime_radius, size_coreset, commu_cost, precision] = ...
    alg_baseline_2(data, k, z, s, mu, epsilon_for_calc_radius)

    tic

    [num_data_total, dim_data] = size(data);
    num_data_site = floor(num_data_total / s);
    
    coresets_ret = cell(1, s);
    weights_ret = cell(1, s);
    coresets = zeros(s*mu*(k+z), dim_data);
    weights = zeros(s*mu*(k+z), 1);
    size_coreset = ones(s, 1) * mu*(k+z);
    commu_cost = s*mu*(k+z)*dim_data + s*mu*(k+z);
    
    precision = 0.0;
    for i = 1:s
        data_idx_left = (i-1) * num_data_site + 1;
        if i == s
            data_idx_right = num_data_total;
        else
            data_idx_right = i * num_data_site;
        end
        [coreset_i, weight_i, precision_i] = gmm(data(data_idx_left:data_idx_right,:), k, z, mu);
        assert(size_coreset(i) == size(coreset_i,1));
        precision = precision + precision_i;
        coresets((i-1)*mu*(k+z)+1:i*mu*(k+z), :) = coreset_i;
        weights((i-1)*mu*(k+z)+1:i*mu*(k+z), :) = weight_i;
        coresets_ret{i} = coreset_i;
        weights_ret{i} = weight_i;
    end
    precision = precision / s;
    if precision > 1.0
        precision = 1.0;
    end
    precision = precision / 6;
    runtime_coreset = toc;
    
    tic
    r_values = pdist(coresets);
    r_values = sort(r_values);
    left_idx = 1;
    right_idx = size(r_values, 2);
    success = 0;
    feasible_centers = [];
    while left_idx <= right_idx
        mid = floor((left_idx + right_idx) / 2);
        [X, weights_T_prime] = outliers_cluster(coresets, weights, k, r_values(:,mid), precision);
        if weights_T_prime <= z
            success = 1;
            feasible_centers = X;
            right_idx = mid - 1;
        else
            left_idx = mid + 1;
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

function [centers, weights, precision] = gmm(data, k, z, mu)
    [num_data, dim_data] = size(data);
    centers = zeros(mu*(k+z), dim_data);
    
    init_center_idx = randi(num_data);
    centers(1, :) = data(init_center_idx, :);
    dist_mat = pdist2(data, centers(1,:));
    
    iter_1 = k + z;
    iter_2 = mu * (k + z);
    for i = 2:iter_1
        [~, nxt_center_idx] = max(dist_mat);
        centers(i, :) = data(nxt_center_idx, :);
        
        tmp_dist_mat = pdist2(data, centers(i,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
    end
    radius_kz = max(dist_mat);
    
    for i = iter_1+1:iter_2
        [~, nxt_center_idx] = max(dist_mat);
        centers(i, :) = data(nxt_center_idx, :);
        
        tmp_dist_mat = pdist2(data, centers(i,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
    end
    radius = max(dist_mat);
    
    weights = zeros(size(centers,1), 1);
    dist_mat = pdist2(data, centers);
    [~, idx] = min(dist_mat, [], 2);
    num_non_centers = size(idx, 1);
    for i = 1:num_non_centers
        weights(idx(i)) = weights(idx(i)) + 1; 
    end
    
    precision = radius / radius_kz * 2;
end

function [centers, weights_T_prime] = outliers_cluster(T, weights, k, r, precision)
    weights_T_prime = sum(weights);
    [num_T, dim_T] = size(T);
    w_temp = zeros(num_T, 1);
    centers = zeros(k, dim_T);
    num_centers = 0;
    % calculate the distances between points in T and T'
    dist_mat = pdist(T);
    dist_mat = squareform(dist_mat);
    is_removed = logical(zeros(num_T, 1));
    while num_centers<k && sum(is_removed)~=num_T
        for i = 1:num_T
            % find the points that can be (1+2*precision)*r covered by T[i]
            condition = dist_mat(i,:)<=(1+2*precision)*r;
            % calculate the weight of T[i]
            w_temp(i, :) = sum(weights(condition, :));
        end
        % select the point with maximum weight
        [~, max_idx] = max(w_temp);
        num_centers = num_centers + 1;
        centers(num_centers, :) = T(max_idx, :);
        % find the points that can be (3+4*precision)*r covered by T[max_idx]
        condition = dist_mat(max_idx,:)<=(3+4*precision)*r;
        % remove these points from T' and w
        is_removed(condition, :) = 1;
        weights_T_prime = weights_T_prime - sum(weights(condition,:));
        weights(condition, :) = 0;
        assert(sum(weights(is_removed,:)) == 0);
    end
    
    if num_centers < k
        for i = 1:num_T
            condition = dist_mat(i,:)<=(1+2*precision)*r;
            w_temp(i, :) = sum(weights(condition, :));
        end
        [~, idx] = maxk(w_temp, k-num_centers);
        centers(num_centers+1:k, :) = T(idx, :);
    end
end
