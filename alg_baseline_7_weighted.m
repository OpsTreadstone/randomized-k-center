% The implementation of the "MK" algorithm of
% Streaming Algorithms for k-Center Clustering with Outliers and with Anonymity

% This is a "weighted" version

function [centers, runtime] = ...
    alg_baseline_7_weighted(coreset, weight, k, z, alpha, beta, eta)
    tic
    
    [num_data, dim_data] = size(coreset);
    num_batch = ceil(sum(weight) / (k*z));
    centers = zeros(k*(z+1), dim_data);
    support_points = cell(1, k*(z+1));
    num_centers = 0;
    
    samples_idx = randperm(num_data, k+z+1);
    sampled_data = coreset(samples_idx, :);
    r = min(pdist(sampled_data)) / 2;
    if r == 0
        r = max(pdist(sampled_data)) / 2;
    end
   
    id = 0;
    idx_lef_next = 1;
    % weighted version
    while (id < num_batch)
        disp(['num_batch=', num2str(num_batch), ', id=', num2str(id)]);
        id = id + 1;
        idx_lef = idx_lef_next;
        idx_rig = idx_lef;
        if id == num_batch
            idx_rig = num_data;
            weight_batch = weight(idx_lef:idx_rig, :);
        else
            total_weight_of_batch = k * z;
            while total_weight_of_batch > 0
                total_weight_of_batch = total_weight_of_batch - weight(idx_rig,:);
                if total_weight_of_batch > 0
                    idx_rig = idx_rig + 1;
                elseif total_weight_of_batch < 0
                    weight_batch = weight(idx_lef:idx_rig, :);
                    weight_batch(idx_rig-idx_lef+1, :) = total_weight_of_batch + weight(idx_rig,:);
                    weight(idx_rig, :) = -total_weight_of_batch;
                    idx_lef_next = idx_rig;
                else
                    weight_batch = weight(idx_lef:idx_rig, :);
                    idx_lef_next = idx_rig + 1;
                end
            end
            assert(sum(weight_batch) == k*z);
        end
        batch = coreset(idx_lef:idx_rig, :);
        batch_size = size(batch, 1);
        
        free_points = logical(ones(batch_size, 1));
        
        success = 0;
        itera = 0;
        num_restart = 0;
        while (~success)
            itera = itera + 1;
            disp(['r = ', num2str(r)]);
%             disp(['id = ', num2str(id), ', itera ', num2str(itera), ...
%                 ', free points = ', num2str(sum(free_points)), ...
%                 ', batch = ', num2str(batch_size), ', centers = ', ...
%                 num2str(num_centers)]);
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
                    % weighted version
                    if sum(weight_batch(close_points,:)) >= z+1
                        num_centers = num_centers + 1;
                        centers(num_centers, :) = batch(i, :);
                        points = batch(close_points, :);
                        % weighted version
                        if sum(weight_batch(close_points,:)) == z+1
                            support_points{num_centers} = points;
                        else
                            weight_close_points = weight_batch(close_points, :);
                            uncovered_weight = z + 1;
                            sps = zeros(z+1, dim_data);
                            num_sps = 0;
                            while uncovered_weight > 0
                                num_sps = num_sps + 1;
                                sps(num_sps, :) = points(num_sps, :);
                                uncovered_weight = uncovered_weight - weight_close_points(num_sps,:);
                            end
                            support_points{num_centers} = sps(1:num_sps, :);
                        end
                        num_changed = 1;
                        break;
                    end
                end
            end
            disp(['After step 1 and 2, centers = ', num2str(num_centers)]);
            
            % weighted version
            if num_centers == k && sum(weight_batch(free_points,:)) <= (k-num_centers+1)*z
                disp('Enter branch-1 and succeed');
                if id == num_batch
                    disp(' ');
                end
                success = 1;
                continue;
            % weighted version
            elseif num_centers < k && sum(weight_batch(free_points,:)) == 0
                if num_restart > 5
                    break;
                end
            % weighted version
            elseif num_centers < k && sum(weight_batch(free_points,:)) <= (k-num_centers+1)*z
                disp(['Enter branch-2 and run base-1, num_free_points: ', num2str(sum(free_points))]);
                [fao_1_centers, fao_1_radius] = four_approx_offline(batch(free_points,:), ...
                    weight_batch(free_points,:), k-num_centers, z);
                assert(size(fao_1_centers,1) == k-num_centers);
                if fao_1_radius <= eta*r
                    success = 1;
                    centers(num_centers+1:k, :) = fao_1_centers;
                    for i = num_centers+1:k
                        dist = pdist2(centers(i,:), batch);
                        close_points = dist<=beta*r;
                        close_points(~free_points) = 0;
                        points = batch(close_points, :);
                        if sum(weight_batch(close_points,:)) == z+1
                            support_points{i} = points;
                        elseif sum(weight_batch(close_points,:)) > z+1
                            weight_close_points = weight_batch(close_points, :);
                            uncovered_weight = z + 1;
                            sps = zeros(z+1, dim_data);
                            num_sps = 0;
                            while uncovered_weight > 0
                                num_sps = num_sps + 1;
                                sps(num_sps, :) = points(num_sps, :);
                                uncovered_weight = uncovered_weight - weight_close_points(num_sps,:);
                            end
                            support_points{i} = sps(1:num_sps, :);
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
                % weighted version
                if num_centers < k && sum(weight_batch(free_points,:)) == 0
                    disp('Failed beacause not enough free points');
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
    
    if num_centers == k
        centers = centers(1:k, :);
    else
        idx = randperm(num_data, k-num_centers);
        centers(num_centers+1:k, :) = coreset(idx, :);
    end
    
    runtime = toc;
end

function [is] = conflict(points_a, points_b, alpha, r)
    if min(pdist2(points_a, points_b)) <= 2*alpha*r
        is = 1;
    else
        is = 0;
    end
end

function [centers, radius] = four_approx_offline(data, weight, k, z)
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
        
        weight_redefined = zeros(num_data, 1);
        for i = 1:num_data
%             disp(['i=', num2str(i), ', size_weight=', ...
%                 num2str(size(weight_redefined,1)), ', size_dist_mat=', ...
%                 num2str(size(dist_mat,1))]);
            % weighted version
            weight_redefined(i, :) = sum(weight(dist_mat(i,:)<=curr_rad,:));
        end
        
        is_removed = zeros(num_data, 1);
        for i = 1:k
            [~, idx_max_weight] = max(weight_redefined);
            centers(i, :) = data(idx_max_weight, :);
            num_centers = num_centers + 1;
            
            idx_to_remove = dist_mat(idx_max_weight,:)<=3*curr_rad;
            is_removed(idx_to_remove, :) = 1;
            dist_mat(idx_to_remove, :) = Inf;
            dist_mat(:, idx_to_remove) = Inf;
            weight_redefined(idx_to_remove, :) = -1;
            
            for j = 1:num_data
                if is_removed(j, :) == 1
                    continue;
                end
                % weighted version
                weight_redefined(j, :) = sum(weight(dist_mat(j,:)<=curr_rad,:));
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
