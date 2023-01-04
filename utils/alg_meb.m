function [center, radius] = alg_meb(data, epsilon)
    num_data = size(data, 1);
    % disp(['Number of data points: ', num2str(num_data)]);
    
    init_center_idx = randi(num_data);
    center = data(init_center_idx, :);
    
    num_iterations = round(1 / epsilon / epsilon);
    % disp(['Number of iterations: ', num2str(num_iterations)]);
    for i = 1:num_iterations
        % disp(['Iteration ', num2str(i)]);
        dist_mat = pdist2(data, center);
        [~, farthest_idx] = max(dist_mat, [], 1);
        new_center = center + (data(farthest_idx,:)-center) / (i+1);
        if pdist2(center, new_center) < 0.001
            center = new_center;
            break
        end
        center = new_center;
    end
    
    dist_mat = pdist2(data, center);
    radius = max(dist_mat, [], 1);
end
