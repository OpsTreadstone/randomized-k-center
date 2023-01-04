function [centers, non_centers, radius] = alg_1_for_alg5(data, z, epsilon, eta, t, mu)
    [num_data, dim_data] = size(data);
    gamma = z / num_data;
    
    num_centers_init = round(1 / (1-gamma) * log(1/eta));
    num_points_iter = round((1+epsilon) * z);
    num_centers_iter = round((1+epsilon) / epsilon * log(1/eta));
    centers = zeros(num_data, dim_data);
    
    is_center = logical(zeros(num_data, 1));
    
    init_centers_idx = randperm(num_data, num_centers_init);
    centers(1:num_centers_init, :) = data(init_centers_idx, :);
    is_center(init_centers_idx, :) = 1;
    
    num_centers_sofar = num_centers_init;
    dist_mat = pdist2(data, centers(1:num_centers_init,:));
    dist_mat = min(dist_mat, [], 2);
    
    for j = 2:t
        [~, Qj_idx] = maxk(dist_mat, num_points_iter);
        idx_in_Qj = randperm(num_points_iter, num_centers_iter);
        centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter, :) = data(Qj_idx(idx_in_Qj), :);
        
        is_center(Qj_idx(idx_in_Qj), :) = 1;
        
        tmp_dist_mat = pdist2(data, centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter,:));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
        
        num_centers_sofar = num_centers_sofar + num_centers_iter;
    end
    
    num_to_exclude = round((1+epsilon) * z);
    [~, idx] = maxk(dist_mat, num_to_exclude+1);
    r_tilde = dist_mat(idx(num_to_exclude + 1));
    
    radius = r_tilde;
    
    num_to_exclude = round((1+epsilon) * 3 * z);
    num_points_iter = round((1+epsilon) * 3*z);
    while radius > mu / 2 * r_tilde
        [~, Qj_idx] = maxk(dist_mat, num_points_iter);
        idx_in_Qj = randperm(num_points_iter, num_centers_iter);
        while sum(is_center(Qj_idx(idx_in_Qj), :)) ~= 0
            idx_in_Qj = randperm(num_points_iter, num_centers_iter);
        end
        centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter, :) = data(Qj_idx(idx_in_Qj), :);
        
        is_center(Qj_idx(idx_in_Qj), :) = 1;
        
        tmp_dist_mat = pdist2(data, centers(num_centers_sofar+1:num_centers_sofar+num_centers_iter, :));
        dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
        
        num_centers_sofar = num_centers_sofar + num_centers_iter;
        
        [~, idx] = maxk(dist_mat, num_to_exclude+1);
        radius = dist_mat(idx(num_to_exclude + 1));
    end
    
    centers = centers(1:num_centers_sofar, :);
    non_centers = data(~is_center, :);
    
    % disp([num2str(size(centers,1)), ', ', num2str(size(non_centers,1)), ', ', num2str(num_data)])
    % disp([num2str(num_centers_sofar), ', ', num2str(sum(is_center)), ', ', num2str(num_data)])
    assert(size(centers,1) + size(non_centers,1) == num_data);
end
