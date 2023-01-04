% The implementation of the "BVX" algorithm of
% Greedy Sampling for Approximate Clustering in the Presence of Outliers

function [centers, radius_z, radius_1_eps_z, runtime] = alg_baseline_1(data, k, z, epsilon_for_calc_radius)
    tic

    [num_data, dim_data] = size(data);
    centers = zeros(k, dim_data);
    
    rand_idx = randi(num_data);
    dist_mat = pdist2(data, data(rand_idx,:));
    lower_bound = 0;
    upper_bound = max(dist_mat);
    
    better_centers = [];
    better_radius = Inf;
    success = 0;
    while ~success
        num_centers = 0;
        r = (lower_bound + upper_bound) / 2;
        
        init_center_idx = randi(num_data);
        centers(1, :) = data(init_center_idx, :);
        num_centers = num_centers + 1;
        dist_mat = pdist2(data, centers(1,:));

        for i = 2:k
            condition = dist_mat>2*r;
            if sum(condition) == 0
                break;
            end
            
            points_2r_far_away = data(condition, :);
            next_center_idx = randi(size(points_2r_far_away, 1));
            centers(i, :) = points_2r_far_away(next_center_idx, :);
            num_centers = num_centers + 1;

            tmp_dist_mat = pdist2(data, centers(i,:));
            dist_mat = min([dist_mat, tmp_dist_mat], [], 2);
        end
        
        outliers = sum(dist_mat>2*r);
        if outliers > z
            if upper_bound-lower_bound <= 0.01
                break;
            else
                lower_bound = r;
            end
        else
            if num_centers < k
                if r < better_radius
                    idx = randperm(num_data, k-num_centers);
                    centers(num_centers+1:k, :) = data(idx, :);
                    better_radius = r;
                    better_centers = centers;
                end
                
                if upper_bound-lower_bound <= 0.01
                    break;
                else
                    upper_bound = r;
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
        radius_z = dist_mat(idx(z+1));

        num_to_remove = round((1+epsilon_for_calc_radius)*z);
        [~, idx] = maxk(dist_mat, num_to_remove+1);
        radius_1_eps_z = dist_mat(idx(num_to_remove+1));
    else
        disp("No need to calculate radius");
        radius_z = -1.0;
        radius_1_eps_z = -1.0;
    end
end
