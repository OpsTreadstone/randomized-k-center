% The implementation of Algorithm 3

function [centers, radius_z, radius_1_eps_z, runtime] = alg_3(data, k, z, epsilon, eta)
    tic
    
    [num_data, dim_data] = size(data);
    gamma = z / num_data;
    sigma = 2 / (1 + sqrt(1 + 4*(1+epsilon)/3/epsilon));
    n_prime = round(3 / sigma / sigma / (1+epsilon) / gamma * log(4/eta));
    t = round((k+sqrt(k)) / (1-eta));
    
    num_centers_init = round(1 / (1-gamma) * log(1/eta));
    num_centers_iter = round((1+sigma) * (1+epsilon) * gamma * n_prime);
    num_centers_total = num_centers_init + (t-1) * num_centers_iter;
    centers = zeros(num_centers_total, dim_data);
    
    init_centers_idx = randperm(num_data, num_centers_init);
    centers(1:num_centers_init, :) = data(init_centers_idx, :);
    num_centers_sofar = num_centers_init;
    
    for j = 2:t
        Aj_idx = randperm(num_data, n_prime);
        dist_mat = pdist2(data(Aj_idx,:), centers(1:num_centers_sofar,:));
        dist_mat = min(dist_mat, [], 2);
        
        [~, Aj_hat_idx] = maxk(dist_mat, num_centers_iter);
        centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter, :) = data(Aj_idx(Aj_hat_idx), :);
        
        num_centers_sofar = num_centers_sofar + num_centers_iter;
    end
    
    runtime = toc;
    
    dist_mat = pdist2(data, centers);
    dist_mat = min(dist_mat, [], 2);
    
    [~, idx] = maxk(dist_mat, z+1);
    radius_z = dist_mat(idx(z+1));
    
    num_to_exclude = round((1+epsilon)*z);
    [~, idx] = maxk(dist_mat, num_to_exclude+1);
    radius_1_eps_z = dist_mat(idx(num_to_exclude + 1));
end

