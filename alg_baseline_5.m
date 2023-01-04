% The implementation of the "LG" algorithm of
% Distributed k-Clustering for Data with Heavy Noise

function [coresets, weights, centers, radius_z, radius_1_eps_z, ...
    runtime_coreset, runtime_radius, size_coreset, commu_cost] = ...
    alg_baseline_5(data, k, z, m, epsilon, epsilon_for_calc_radius)
    tic
    
    [num_data, dim_data] = size(data);
    num_data_machine = floor(num_data / m);
    coresets = cell(1, m);
    weights = cell(1, m);
    commu_cost = 0;
    
    rand_idx = randi(num_data);
    dist_mat = pdist2(data, data(rand_idx,:));
    lower_bound = 0;
    upper_bound = max(dist_mat);
    total_centers_at_most = k * m * (1 + 1/epsilon);
    total_covered_at_least = max([num_data-(1+epsilon)*z, 1]);
    
    success = 0;
    while upper_bound-lower_bound >= 0.01
        L = (lower_bound + upper_bound) / 2;
        total_centers = 0;
        total_covered = 0;
        for i = 1:m
            data_idx_left = (i-1) * num_data_machine + 1;
            if i == m
                data_idx_right = num_data;
            else
                data_idx_right = i * num_data_machine;
            end
            [coresets{i}, weights{i}, num_centers, num_covered] = ...
                aggregating(data(data_idx_left:data_idx_right,:), L, epsilon*z/k/m);
            total_centers = total_centers + num_centers;
            total_covered = total_covered + num_covered;
        end
        
        commu_cost = commu_cost + m;
        
        if total_centers>total_centers_at_most || total_covered<total_covered_at_least
            lower_bound = L;
        else
            success = 1;
            feasible_coresets = coresets;
            feasible_weights = weights;
            upper_bound = L;
        end
    end
    assert(success==1);
    coresets = feasible_coresets;
    weights = feasible_weights;
    
    runtime_coreset = toc;
    
    coresets_compact = zeros(num_data, dim_data);
    weights_compact = zeros(num_data, 1);
    size_coreset = zeros(m, 1);
    num_coreset = 0;
    for i = 1:m
        size_coreset(i) = size(coresets{i}, 1);
        coresets_compact(num_coreset+1:num_coreset+size_coreset(i), :) = coresets{i};
        weights_compact(num_coreset+1:num_coreset+size_coreset(i), :) = weights{i};
        num_coreset = num_coreset + size_coreset(i);
    end
    coresets_compact = coresets_compact(1:num_coreset, :);
    weights_compact = weights_compact(1:num_coreset, :);
    commu_cost = num_coreset*dim_data + num_coreset;
    tic
    centers = kzc(coresets_compact, weights_compact, k, 5*upper_bound);
    runtime_radius = toc;
    
    dist_mat = pdist2(data, centers);
    dist_mat = min(dist_mat, [], 2);
    [~, idx] = maxk(dist_mat, z+1);
    radius_z = dist_mat(idx(z+1));
    num_to_remove = round((1+epsilon_for_calc_radius)*z);
    [~, idx] = maxk(dist_mat, num_to_remove+1);
    radius_1_eps_z = dist_mat(idx(num_to_remove + 1));
end

function [Q_prime, w_prime, num_Q_prime, num_covered] = aggregating(Q, L, y)
    [num_Q, dim_Q] = size(Q);
    Q_prime = zeros(num_Q, dim_Q);
    w_prime = zeros(num_Q, 1);
    num_Q_prime = 0;

    is_removed = logical(zeros(num_Q, 1));
    for i = 1:num_Q
        dist = pdist2(Q(i,:), Q(~is_removed,:));
        if sum(dist<=2*L) > y
            num_Q_prime = num_Q_prime + 1;
            Q_prime(num_Q_prime, :) = Q(i, :);
            tmp_dist = pdist2(Q(i,:), Q);
            condition = tmp_dist<=4*L;
            condition(is_removed) = 0;
            w_prime(num_Q_prime, :) = sum(condition);
            is_removed(condition, :) = 1;
        end
    end
    
    Q_prime = Q_prime(1:num_Q_prime, :);
    w_prime = w_prime(1:num_Q_prime, :);
    num_covered = num_Q - sum(~is_removed);
    
    assert(sum(w_prime) == num_covered);
end

function C_prime = kzc(Q, w_prime, k, L_prime)
    [num_Q, dim_Q] = size(Q);
    C_prime = zeros(k, dim_Q);
    is_removed = logical(zeros(num_Q, 1));
    for i = 1:k
        U = Q(~is_removed, :);
        w_U = w_prime(~is_removed, :);
        w_Q = zeros(num_Q, 1);
        for j = 1:num_Q
            dist = pdist2(Q(j,:), U);
            condition = dist<=2*L_prime;
            w_Q(j, :) = sum(w_U(condition,:));
        end
        
        [~, idx] = max(w_Q);
        C_prime(i, :) = Q(idx, :);

        if sum(is_removed)==num_Q && i<k
            [~, idx] = maxk(w_Q, k-i+1);
            size_idx = size(idx, 1);
            if size_idx == k-i+1
                C_prime(i:k, :) = Q(idx, :);
            else
                C_prime(i:i+size_idx-1, :) = Q(idx, :);
                C_prime(i+size_idx:k, :) = C_prime(1:k-i-size_idx+1, :);
            end
            break;
        end
        
        tmp_dist = pdist2(C_prime(i,:), Q);
        condition = tmp_dist<=4*L_prime;
        % condition(is_removed) = 0;
        is_removed(condition, :) = 1;
    end
end
