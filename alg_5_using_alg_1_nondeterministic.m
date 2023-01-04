% The implementation of Algorithm 5

function [coreset, weight, runtime, radius] = alg_5_using_alg_1_nondeterministic(data, k, z, delta, mu)
    tic
    
    epsilon = 1;
    eta = delta / 2;
    
    dim_data = size(data, 2);
    p = 2 + (2 / k / (1-eta)) * log(2/delta);
    t = round(p * k / (1-eta));
    
    num_to_exclude = round((1+epsilon) * 3 * z);
    
    [coreset_alg_1, others, radius] = alg_1_for_alg5(data, z, epsilon, eta, t, mu);
    size_coreset_alg_1 = size(coreset_alg_1, 1);
    
    % assert(size_coreset_alg_1+size(others,1) == size(data,1));
    
    dist_mat = pdist2(others, coreset_alg_1);
    [dist_mat, idx] = min(dist_mat, [], 2);
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
    
    runtime = toc;
end
