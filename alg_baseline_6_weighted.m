% The implementation of the "CKM+" algorithm of
% Algorithms for Facility Location Problems with Outliers

% This is a "weighted" version

function [centers, runtime] = alg_baseline_6_weighted(coreset, weight, k, z)
    disp("Using base 6");

    tic

    [num_data, dim_data] = size(coreset);
    try
        dist_mat_whole_data = pdist(coreset);
        min_dist = min(dist_mat_whole_data);
        max_dist = max(dist_mat_whole_data);
        dist_mat_whole_data = squareform(dist_mat_whole_data);
        out_of_memory = 0;
    catch
        out_of_memory = 1;
        min_dist = Inf;
        max_dist = -1;
        for i = 1:size(coreset,1)
            temp_dist_mat = pdist2(coreset, coreset(i,:));
            min_dist = min([min_dist, min(temp_dist_mat)]);
            max_dist = max([max_dist, max(temp_dist_mat)]);
        end
    end
    
    centers = zeros(k, dim_data);
    
    better_centers = [];
    success = 0;
    while (~success)
        if out_of_memory == 0
            dist_mat = dist_mat_whole_data;
        else
            idx_to_remove = logical(zeros(size(coreset,1), 1));
        end
%         disp(['min_dist=', num2str(min_dist), ', max_dist=', ...
%             num2str(max_dist)]);
        if min_dist > max_dist
            break;
        end
        
        num_centers = 0;
        curr_rad = (min_dist + max_dist) / 2;
        
        % weighted version
        weight_redefined = zeros(num_data, 1);
        for i = 1:num_data
            if out_of_memory == 0
                weight_redefined(i, :) = sum(weight(dist_mat(i,:)<=curr_rad,:));
            else
                temp_dist_mat = pdist2(coreset(i,:), coreset);
                weight_redefined(i, :) = sum(weight(temp_dist_mat<=curr_rad,:));
            end
        end
        
        is_removed = zeros(num_data, 1);
        for i = 1:k
            [~, idx_max_weight] = max(weight_redefined);
            centers(i, :) = coreset(idx_max_weight, :);
            num_centers = num_centers + 1;
            
            if out_of_memory == 0
                idx_to_remove = dist_mat(idx_max_weight,:)<=3*curr_rad;
                dist_mat(idx_to_remove, :) = Inf;
                dist_mat(:, idx_to_remove) = Inf;
            else
                temp_dist_mat = pdist2(coreset(idx_max_weight,:), coreset);
                idx_to_remove = bitor(temp_dist_mat<=3*curr_rad, idx_to_remove);
            end
            is_removed(idx_to_remove, :) = 1;
            weight_redefined(idx_to_remove, :) = -1;
            
            for j = 1:num_data
                if is_removed(j, :) == 1
                    continue;
                end
                % weighted version
                if out_of_memory == 0
                    weight_redefined(j, :) = sum(weight(dist_mat(j,:)<=curr_rad,:));
                else
                    temp_dist_mat = pdist2(coreset(j,:), coreset);
                    temp_dist_mat(:, idx_to_remove) = Inf;
                    weight_redefined(j, :) = sum(weight(temp_dist_mat<=curr_rad,:));
                end
            end
            
            if sum(~is_removed) == 0
                break;
            end
        end
        
        % weighted version
        curr_outliers = sum(weight(~is_removed,:));
        if curr_outliers > z
            if max_dist-min_dist < 0.01
                break;
            end
            min_dist = curr_rad;
        else
            if num_centers < k
                [~, idx] = maxk(weight_redefined, k-num_centers);
                size_idx = size(idx, 1);
                if size_idx == k-num_centers
                    centers(num_centers+1:k, :) = coreset(idx, :);
                else
                    centers(num_centers+1:num_centers+size_idx, :) = coreset(idx, :);
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
end
