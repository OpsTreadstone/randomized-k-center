% The implementation of the "GLZ" algorithm of
% Distributed Partial Clustering

function [coresets, weights, centers, radius_z, radius_1_eps_z, ...
    runtime_coreset, runtime_radius, size_coreset, commu_cost] = ...
    alg_baseline_4(data, k, z, s, rho, epsilon_for_calc_radius)

    tic
    [num_data, dim_data] = size(data);
    num_data_site = floor(num_data / s);
    
    commu_cost = 0;

    el = zeros(s, z);
    
    for i = 1:s
        data_idx_left = (i-1) * num_data_site + 1;
        if i == s
            data_idx_right = num_data;
        else
            data_idx_right = i * num_data_site;
        end
        curr_data = data(data_idx_left:data_idx_right, :);
        
        re_ordering = alg_gonzalez_for_base4(curr_data, k+z);
        for q = 1:z
            dist = pdist2(re_ordering(k+q,:), re_ordering(1:k+q-1,:));
            el(i, q) = min(dist);
        end
    end
    commu_cost = commu_cost + s*z;
    
    el_temp = el';
    [el_sorted_val, el_sorted_idx] = sort(el_temp(:), 'descend');
    rank = rho * z;
    el_i0_q0 = el_sorted_val(rank);
    i0 = ceil(el_sorted_idx(rank) / z);
    if mod(el_sorted_idx(rank), z) == 0
        q0 = z;
    else
        q0 = mod(el_sorted_idx(rank), z);
    end
    commu_cost = commu_cost + s*3;
    
    coresets = cell(1, s);
    weights = cell(1, s);
    size_coreset = zeros(s, 1);
    for i = 1:s
        data_idx_left = (i-1) * num_data_site + 1;
        if i == s
            data_idx_right = num_data;
        else
            data_idx_right = i * num_data_site;
        end
        curr_data = data(data_idx_left:data_idx_right, :);
        num_curr_data = size(curr_data, 1);
        
        q_max = -1;
        if i == i0
            q_max = q0;
        else
            for q = 1:z
                if el(i,q) >= el_i0_q0
                    q_max = q;
                end
            end
        end
        assert(q_max > -1);
        
        coresets{i} = alg_gonzalez_for_base4(curr_data, 2*k+q_max);
        size_coreset(i,:) = size(coresets{i}, 1);
        
        dist_mat = pdist2(curr_data, coresets{i});
        [~, idx] = min(dist_mat, [], 2);
        
        weight_i = zeros(size_coreset(i,:), 1);
        for j = 1:num_curr_data
            weight_i(idx(j)) = weight_i(idx(j)) + 1;
        end
        assert(sum(weight_i) == num_curr_data);
        weights{i} = weight_i;
        
        commu_cost = commu_cost + size_coreset(i,:)*dim_data + size_coreset(i,:);
    end
    runtime_coreset = toc;
    
    coresets_compact = zeros(num_data, dim_data);
    weights_compact = zeros(num_data, 1);
    idx_a = 1;
    for i = 1:s
        idx_b = idx_a + size(coresets{i},1) - 1;
        coresets_compact(idx_a:idx_b, :) = coresets{i};
        weights_compact(idx_a:idx_b, :) = weights{i};
        idx_a = idx_b + 1;
    end
    coresets_compact = coresets_compact(1:idx_b, :);
    weights_compact = weights_compact(1:idx_b, :);
    assert(idx_b == sum(size_coreset));
    assert(sum(weights_compact) == num_data);
    
    [centers, runtime_radius] = alg_baseline_6_weighted(coresets_compact, weights_compact, k, z);
    
    dist_mat = pdist2(data, centers);
    dist_mat = min(dist_mat, [], 2);
    [~, idx] = maxk(dist_mat, z+1);
    radius_z = dist_mat(idx(z+1));
    num_to_remove = round((1+epsilon_for_calc_radius)*z);
    [~, idx] = maxk(dist_mat, num_to_remove+1);
    radius_1_eps_z = dist_mat(idx(num_to_remove + 1));
end

function centers = alg_gonzalez_for_base4(data, k)
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
