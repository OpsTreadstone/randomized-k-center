% The implementation of Algorithm 6

function [coresets, weights, runtime, size_coresets_machine, commu_cost] = alg_6(data, ...
    k, z, s, mu, delta, epsilon)

    tic
    eta = delta / 2;
    varrho = 2;
    
    [num_data_total, dim_data] = size(data);
    num_data_site = floor(num_data_total / s);
    
    r_max = floor(log(z) / log(varrho));
    if r_max < log(z)/log(varrho)
        q_size = r_max + 2;
    else
        q_size = r_max + 1;
    end
    q_values = zeros(q_size, 1);
    q_values(2) = varrho;
    q_values(q_size) = z;
    for i = 3:q_size-1
        q_values(i) = q_values(i-1) * varrho;
    end
    % commu_cost = 0 + (f in step 2) + (f(i0,q0) in step 3)
    commu_cost = s*q_size + s;

    f = zeros(s, z+1);
    coresets_bank = cell(s, q_size);
    weights_bank = cell(s, q_size);
    
    for i = 1:s
        data_idx_left = (i-1) * num_data_site + 1;
        if i == s
            data_idx_right = num_data_total;
        else
            data_idx_right = i * num_data_site;
        end
        
        for q_idx = 1:q_size
            q = q_values(q_idx);
            num_points_iter = ceil((1+epsilon) * q);
            num_centers_iter = ceil((1+epsilon) / epsilon * log(1/eta));
            if num_points_iter < num_centers_iter
                f(i,q+1) = Inf;
            elseif data_idx_right-data_idx_left+1 < num_points_iter
                f(i,q+1) = Inf;
            else
                [coresets_bank{i,q_idx}, weights_bank{i,q_idx}, ~, f(i,q+1)] = alg_5_using_alg_1_nondeterministic(data(data_idx_left:data_idx_right,:), ...
                    k, q, delta, mu);
            end
            
            if q_idx > 1
                prev_q = q_values(q_idx-1);
                if f(i,q+1) > f(i,prev_q+1)
                    f(i,q+1) = f(i,prev_q+1);
                end
            end
            
            if q_idx < q_size
                q_nxt = q_values(q_idx+1) - 1;                  
                for j = q+1:q_nxt
                    f(i,j+1) = f(i,q+1);
                end
            end
        end
    end
    
    f_temp = f';
    [f_sorted_val, f_sorted_idx] = sort(f_temp(:), 'descend');
    rank = varrho * z + 1;
    f_i0_q0 = f_sorted_val(rank);
%     i0 = ceil(f_sorted_idx(rank) / q_size);
%     if mod(f_sorted_idx(rank), q_size) == 0
%         q0 = q_size;
%     else
%         q0 = mod(f_sorted_idx(rank), q_size);
%     end
    
    coresets = cell(1, s);
    weights = cell(1, s);
    size_coresets_machine = zeros(s, 1);
    for i = 1:s
%         data_idx_left = (i-1) * num_data_site + 1;
%         if i == s
%             data_idx_right = num_data_total;
%         else
%             data_idx_right = i * num_data_site;
%         end
        
        q_min = Inf;
        for q_idx = 1:q_size
            if q_values(q_idx) < q_min && f(i,q_values(q_idx)+1) <= f_i0_q0
                if q_values(q_idx) == 0
                    continue;
                else
                    q_min = q_values(q_idx);
                    q_min_idx = q_idx;
                    break;
                end
            end
        end
        % assert(q_min < Inf);
        if q_min == Inf
            q_min = z;
            q_min_idx = q_size;
        end
        
        coresets{i} = coresets_bank{i, q_min_idx};
        weights{i} = weights_bank{i, q_min_idx};
        % [coresets{i}, weights{i}, ~, ~] = alg_5_using_alg_1_nondeterministic(data(data_idx_left:data_idx_right,:), ...
        %     k, q_min, delta, mu);
        size_coresets_machine(i, :) = size(coresets{i}, 1);
        % commu_cost += (Ei and associated weight in step 4(d))
        commu_cost = commu_cost + size_coresets_machine(i,:)*dim_data + size_coresets_machine(i,:);
    end
    runtime = toc;
end
