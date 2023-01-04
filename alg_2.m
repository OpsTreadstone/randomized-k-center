% The implementation of Algorithm 2

function [centers, radius_z, radius_1_eps_z, runtime] = alg_2(data, k, z, epsilon)
    tic
    
    [num_data, dim_data] = size(data);
    num_points_iter = round((1+epsilon) * z);
    centers = zeros(k, dim_data);
    
    init_center_idx = randi(num_data);
    centers(1, :) = data(init_center_idx, :);
    dist_mat = pdist2(data, centers(1,:));
    
    for i = 2:k
        [~, Qj_idx] = maxk(dist_mat, num_points_iter);
        idx_in_Qj = randi(num_points_iter);
        centers(i, :) = data(Qj_idx(idx_in_Qj),:);
        
        tmp_dist_mat = pdist2(data, centers(i,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
    end
    
    runtime = toc;
    
    [~, idx] = maxk(dist_mat, z+1);
    radius_z = dist_mat(idx(z+1));
    
    [~, idx] = maxk(dist_mat, num_points_iter+1);
    radius_1_eps_z = dist_mat(idx(num_points_iter+1));
end
