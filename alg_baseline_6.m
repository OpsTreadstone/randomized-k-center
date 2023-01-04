% The implementation of the "CKM+" algorithm of
% Algorithms for Facility Location Problems with Outliers

function [centers, radius_z, radius_1_eps_z, runtime] = ...
    alg_baseline_6(data, k, z, epsilon_for_calc_radius)
    tic

    [num_data, dim_data] = size(data);
    dist_mat_whole_data = pdist(data);
    min_dist = min(dist_mat_whole_data);
    max_dist = max(dist_mat_whole_data);
    dist_mat_whole_data = squareform(dist_mat_whole_data);
    
    centers = zeros(k, dim_data);
    
    better_centers = [];
    success = 0;
    while (~success)
        dist_mat = dist_mat_whole_data;
%         disp(['min_dist=', num2str(min_dist), ', max_dist=', ...
%             num2str(max_dist)]);
        if min_dist > max_dist
            break;
        end
        
        num_centers = 0;
        curr_rad = (min_dist + max_dist) / 2;
        
        weight = zeros(num_data, 1);
        for i = 1:num_data
            weight(i, :) = sum(dist_mat(i,:) <= curr_rad);
        end
        
        is_removed = zeros(num_data, 1);
        for i = 1:k
            [~, idx_max_weight] = max(weight);
            centers(i, :) = data(idx_max_weight, :);
            num_centers = num_centers + 1;
            
            idx_to_remove = dist_mat(idx_max_weight,:)<=3*curr_rad;
            is_removed(idx_to_remove, :) = 1;
            dist_mat(idx_to_remove, :) = Inf;
            dist_mat(:, idx_to_remove) = Inf;
            weight(idx_to_remove, :) = -1;
            
            for j = 1:num_data
                if is_removed(j, :) == 1
                    continue;
                end
                weight(j, :) = sum(dist_mat(j,:) <= curr_rad);
            end
            
            if sum(~is_removed) == 0
                break;
            end
        end
        
        curr_outliers = sum(~is_removed);
        if curr_outliers > z
            if max_dist-min_dist < 0.01
                break;
            end
            min_dist = curr_rad;
        else
            if num_centers < k
                [~, idx] = maxk(weight, k-num_centers);
                size_idx = size(idx, 1);
                if size_idx == k-num_centers
                    centers(num_centers+1:k, :) = data(idx, :);
                else
                    centers(num_centers+1:num_centers+size_idx, :) = data(idx, :);
                    centers(num_centers+size_idx+1:k, :) = centers(1:k-num_centers-size_idx, :);
                end
                better_centers = centers;
                
                if max_dist-min_dist < 0.01
                    break;
                else
                    max_dist = curr_rad;
                end
            else
                success = 1;
            end
        end
    end
    assert(~isempty(better_centers) || success);
    
    if ~success
        centers = better_centers;
    end
    
    runtime = toc;
    
    if exist('epsilon_for_calc_radius', 'var')
        dist_mat = pdist2(data, centers);
        dist_mat = min(dist_mat, [], 2);

        [~, idx] = maxk(dist_mat, z+1);
        radius_z = dist_mat(idx(z + 1));

        num_to_remove = round((1+epsilon_for_calc_radius)*z);
        [~, idx] = maxk(dist_mat, num_to_remove+1);
        radius_1_eps_z = dist_mat(idx(num_to_remove + 1));
    else
        disp("No need to calculate radius");
        radius_z = -1.0;
        radius_1_eps_z = -1.0;
    end
end
