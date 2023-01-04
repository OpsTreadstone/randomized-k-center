% The implementation of the "UNIFORM" algorithm

function [coreset, weight, runtime] = alg_uniform(data, k)
    tic
    
    num_data = size(data, 1);
    coreset_idx = randperm(num_data, k);
    coreset = data(coreset_idx, :);

    weight = ones(k, 1);
    
    runtime = toc;
end