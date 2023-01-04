% The implementation of the "MK" algorithm of
% Streaming Algorithms for k-Center Clustering with Outliers and with Anonymity

function [centers, radius_z, radius_1_eps_z, runtime] = ...
    alg_baseline_7(data, k, z, alpha, beta, eta, epsilon_for_calc_radius)
    tic
    
    [num_data, dim_data] = size(data);
    num_batch = ceil(num_data / (k*z));
    centers = zeros(k*(z+1), dim_data);
    support_points = cell(1, k*(z+1));
    num_centers = 0;
    
    samples_idx = randperm(num_data, k+z+1);
    sampled_data = data(samples_idx, :);
    r = min(pdist(sampled_data)) / 2;
    if r == 0
        r = max(pdist(sampled_data)) / 2;
    end
   
    id = 0;
    while (id < num_batch)
        idx_lef = id * k * z + 1;
        id = id + 1;
        if id == num_batch
            idx_rig = num_data;
        else
            idx_rig = id * k * z;
        end
        batch = data(idx_lef:idx_rig, :);
        batch_size = size(batch, 1);
        
        free_points = logical(ones(batch_size, 1));

        success = 0;
        itera = 0;
        num_restart = 0;
        while (~success)
            itera = itera + 1;
            disp(['id = ', num2str(id), ', itera ', num2str(itera), ...
                ', free points = ', num2str(sum(free_points)), ...
                ', batch = ', num2str(batch_size), ', centers = ', ...
                num2str(num_centers)]);
            prev_num_centers = num_centers;
            num_changed = 1;
            while num_changed == 1
                num_changed = 0;
                
                if num_centers ~=0
                    dist_mat = pdist2(batch, centers(1:num_centers,:));
                    dist_mat = min(dist_mat, [], 2);
                    drop = dist_mat<=eta*r;
                    free_points(drop, :) = 0;
                end
                
                for i = 1:batch_size
                    if ~free_points(i,:)
                        continue;
                    end
                    dist = pdist2(batch(i,:), batch);
                    close_points = dist<=beta*r;
                    close_points(~free_points) = 0;
                    if sum(close_points) >= z+1
                        num_centers = num_centers + 1;
                        centers(num_centers, :) = batch(i, :);
                        points = batch(close_points, :);
                        support_points{num_centers} = points(1:z+1, :);
                        num_changed = 1;
                        break;
                    end
                end
            end
            disp(['After step 1 and 2, centers = ', num2str(num_centers)]);
            
            if num_centers == k && sum(free_points) <= (k-num_centers+1)*z
                disp('Enter branch-1 and succeed');
                if id == num_batch
                    disp(' ');
                end
                success = 1;
                continue;
            elseif num_centers < k && sum(free_points) == 0
                if num_restart > 5
                    break;
                end
            elseif num_centers < k && sum(free_points) <= (k-num_centers+1)*z
                disp(['Enter branch-2 and run base-1, num_free_points: ', num2str(sum(free_points))]);
                [fao_1_centers, fao_1_radius] = four_approx_offline(batch(free_points,:), k-num_centers, z);
                assert(size(fao_1_centers,1) == k-num_centers);
                if fao_1_radius <= eta*r
                    success = 1;
                    centers(num_centers+1:k, :) = fao_1_centers;
                    for i = num_centers+1:k
                        dist = pdist2(centers(i,:), batch);
                        close_points = dist<=beta*r;
                        close_points(~free_points) = 0;
                        points = batch(close_points, :);
                        if sum(close_points) >= z+1
                            support_points{i} = points(1:z+1, :);
                        else
                            support_points{i} = centers(i, :);
                        end
                    end
                    num_centers = k;
                else
                    disp('The resulting radius is too large');
                end
            else
                if num_centers > k
                    disp(['Enter branch-3 because num of centers is ', ...
                        num2str(num_centers), ' > ', num2str(k)]);
                else
                    disp('Enter branch-3 because of too many free points');
                end
            end
            if success == 0
                if num_centers < k && sum(free_points) == 0
                    disp('Failed beacause not enough free points');
%                     samples_idx = randperm(num_data, k+z+1);
%                     sampled_data = data(samples_idx, :);
%                     r = min(pdist(sampled_data)) / 2;
                    r = r / 1.5;
                    free_points = logical(ones(batch_size, 1));
                    num_centers = prev_num_centers;
                    num_restart = num_restart + 1;
                else
                    disp('Failed');
                    r = alpha * r;
                    dropped = zeros(num_centers, 1);
                    curr_num_centers = 0;
                    for i = 1:num_centers
                        if dropped(i, :) == 1
                            continue;
                        end
                        
                        curr_num_centers = curr_num_centers + 1;
                        centers(curr_num_centers, :) = centers(i, :);
                        support_points{curr_num_centers} = support_points{i};
                        
                        for j = i+1:num_centers
                            if conflict(support_points{i}, support_points{j}, alpha, r) == 1
                                dropped(j, :) = 1;
                            end
                        end
                    end
                    num_centers = curr_num_centers;
                end
            else
                disp('Succeed');
                if id == num_batch
                    disp(' ');
                end
            end
        end
    end
    
    runtime = toc;
    
    if exist('epsilon_for_calc_radius', 'var')
        centers = centers(1:k, :);
        dist_mat = pdist2(data, centers);
        dist_mat = min(dist_mat, [], 2);

        [~, idx] = maxk(dist_mat, z + 1);
        radius_z = dist_mat(idx(z + 1));

        num_to_remove = round((1+epsilon_for_calc_radius)*z);
        [~, idx] = maxk(dist_mat, num_to_remove + 1);
        radius_1_eps_z = dist_mat(idx(num_to_remove + 1));
    else
        disp("No need to calculate radius");
        radius_z = -1.0;
        radius_1_eps_z = -1.0;
    end
end

function [is] = conflict(points_a, points_b, alpha, r)
    if min(pdist2(points_a, points_b)) <= 2*alpha*r
        is = 1;
    else
        is = 0;
    end
end

function [centers, radius] = four_approx_offline(data, k, z)
    [num_data, dim_data] = size(data);
    if num_data == 1
        dist_mat_whole_data = 0;
        min_dist = 0;
        max_dist = 0;
    else
        dist_mat_whole_data = pdist(data);
        min_dist = min(dist_mat_whole_data);
        max_dist = max(dist_mat_whole_data);
        dist_mat_whole_data = squareform(dist_mat_whole_data);
    end
    
    centers = zeros(k, dim_data);
    
    better_centers = [];
    success = 0;
    while (~success)
        dist_mat = dist_mat_whole_data;
%         disp(['size_dist_mat=', num2str(size(dist_mat,1))]);
%         disp(['size_dist_mat_whole_data=', num2str(size(dist_mat_whole_data,1))]);
%         disp(['min_dist=', num2str(min_dist), ', max_dist=', ...
%             num2str(max_dist)]);
        if min_dist > max_dist
            break;
        end
        
        num_centers = 0;
        curr_rad = (min_dist + max_dist) / 2;
        
        weight = zeros(num_data, 1);
        for i = 1:num_data
%             disp(['i=', num2str(i), ', size_weight=', ...
%                 num2str(size(weight,1)), ', size_dist_mat=', ...
%                 num2str(size(dist_mat,1))]);
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
    
    if size(data, 1) < z+1
        radius = 0;
    else
        dist_mat = pdist2(data, centers);
        dist_mat = min(dist_mat, [], 2);
        [~, idx] = maxk(dist_mat, z + 1);
        radius = dist_mat(idx(z + 1));
    end
end
