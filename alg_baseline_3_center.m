% The implementation of the "CLUSTER" algorithm of
% Fast Distributed k-Center Clustering with Outliers on Massive Data

function [centers, runtime] = alg_baseline_3_center(data, coresets, weights, k, z)
    tic
    
    num_data = size(data, 1);
    rand_idx = randi(num_data);
    dist_mat = pdist2(data, data(rand_idx,:));
    lower_bound = 0;
    upper_bound = max(dist_mat);
    
    success = 0;
    feasible_centers = [];
    while upper_bound-lower_bound >= 0.01
        G = (lower_bound + upper_bound) / 2;
        [centers, weight_U_prime] = alg_cluster_(coresets, weights, k, G);
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
    runtime = toc;    
end

function [centers, weight_U_prime] = alg_cluster_(coreset, weight, k, G)
    [num_coreset, dim_data] = size(coreset);
    dist_mat = pdist(coreset);
    dist_mat = squareform(dist_mat);
    
    num_centers = 0;
    centers = zeros(k, dim_data);
    weight_redefined = zeros(num_coreset, 1);
    is_removed = logical(zeros(num_coreset, 1));
    
    while num_centers<k && sum(is_removed)~=num_coreset
        for i = 1:num_coreset
            condition = dist_mat(i,:)<=5*G;
            weight_redefined(i, :) = sum(weight(condition, :));
        end
        
        [~, idx_max_weight] = max(weight_redefined);
        num_centers = num_centers + 1;
        centers(num_centers, :) = coreset(idx_max_weight, :);
        
        condition = dist_mat(idx_max_weight,:)<=11*G;
        is_removed(condition, :) = 1;
        weight(condition, :) = 0;
        assert(sum(weight(is_removed,:)) == 0);
    end
    if sum(is_removed)~=num_coreset
        weight_U_prime = sum(weight(~is_removed));
    else
        weight_U_prime = 0;
    end
    
    if num_centers < k
        for i = 1:num_coreset
            condition = dist_mat(i,:)<=5*G;
            weight_redefined(i, :) = sum(weight(condition, :));
        end
        [~, idx] = maxk(weight_redefined, k-num_centers);
        centers(num_centers+1:k, :) = coreset(idx, :);
    end
end
