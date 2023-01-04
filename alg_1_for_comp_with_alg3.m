% The implementation of Algorithm 1, modified to output the same number of 
% points as Algorithm 3

function [centers, radius_z, radius_1_eps_z, runtime] = alg_1_for_comp_with_alg3(data, z, epsilon, eta, t, target_num)
    tic

    [num_data, dim_data] = size(data);
    gamma = z / num_data;
    
    num_centers_init = round(1 / (1-gamma) * log(1/eta));
    num_points_iter = round((1+epsilon) * z);
    num_centers_iter = round((1+epsilon) / epsilon * log(1/eta));
    centers = zeros(target_num, dim_data);
    
    init_centers_idx = randperm(num_data, num_centers_init);
    centers(1:num_centers_init, :) = data(init_centers_idx, :);
    num_centers_sofar = num_centers_init;
    dist_mat = pdist2(data, centers(1:num_centers_init,:));
    dist_mat = min(dist_mat, [], 2);
    
    for j = 2:t
        [~, Qj_idx] = maxk(dist_mat, num_points_iter);
        idx_in_Qj = randperm(num_points_iter, num_centers_iter);
        centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter, :) = data(Qj_idx(idx_in_Qj), :);
        
        tmp_dist_mat = pdist2(data, centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
        
        num_centers_sofar = num_centers_sofar + num_centers_iter;
    end
    
    if num_centers_sofar < target_num
        [~, Qj_idx] = maxk(dist_mat, num_points_iter);
        idx_in_Qj = randperm(num_points_iter, target_num-num_centers_sofar);
        centers(num_centers_sofar+1:target_num, :) = data(Qj_idx(idx_in_Qj), :);
        
        tmp_dist_mat = pdist2(data, centers(num_centers_sofar+1:target_num,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
    end
    
    runtime = toc;
    
    [~, idx] = maxk(dist_mat, z+1);
    radius_z = dist_mat(idx(z+1));
    
    [~, idx] = maxk(dist_mat, num_points_iter+1);
    radius_1_eps_z = dist_mat(idx(num_points_iter+1));
end
