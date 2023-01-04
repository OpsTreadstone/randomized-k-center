% Randomly add outliers inside 1.1*r_meb
% so some outliers are mixed with inliners

dataset_ = 'shuttle';

if strcmp(dataset_, "cifar10") || strcmp(dataset_, "fashion_mnist") || strcmp(dataset_, "mnist") || strcmp(dataset_, "pokerhand") || strcmp(dataset_, "svhn")
    k = 10;
elseif strcmp(dataset_, "covertype") || strcmp(dataset_, "shuttle") || strcmp(dataset_, "tiny_covertype")
    k = 7;
elseif strcmp(dataset_, "kddcup99")
    k = 23;
elseif strcmp(dataset_, "tiny_kddcup99")
    k = 14;
elseif strcmp(dataset_, "tiny_pokerhand")
    k = 9;
end

disp(['dataset_: ', dataset_, ', k: ', num2str(k)]);
pause(5);

data_folder = ['../datasets/', dataset_];
load([data_folder, '/', dataset_, '.mat']);
[num_data, dim_data] = size(data);

save_folder_outlier = data_folder;

threshold_on_meb_whole = 1.1;

ratio_outliers_values = [0.01];

% calculate r_meb and c_meb of the whole dataset
disp("Calculating MEB of the whole dataset...");
[c_meb_whole, r_meb_whole] = alg_meb(data, 0.01);
file_meb_whole = [data_folder, '/', dataset_, '_meb_whole.mat'];
save(file_meb_whole, 'c_meb_whole', 'r_meb_whole');

% calculate r_i and c_i of each class
[c_meb_class, r_meb_class] = get_meb_class(data, k, data_folder, dataset_);
assert(sum(sum(c_meb_class==zeros(k,dim_data),2)==dim_data) == 0);
assert(sum(r_meb_class==zeros(k,1)) == 0);
file_meb_class = [data_folder, '/', dataset_, '_meb_class.mat'];
save(file_meb_class, 'c_meb_class', 'r_meb_class');

% generate points in the ball of radius threshold_on_meb_whole*r_meb centered at c_meb
max_radi = threshold_on_meb_whole * r_meb_whole;
for i = 1:size(ratio_outliers_values, 2)
    disp(['ratio_outliers: ', num2str(ratio_outliers_values(i))]);
    
    ratio_outliers = ratio_outliers_values(i);
    num_outliers = round(num_data * ratio_outliers);
    
    random_directions = randn(num_outliers, dim_data);
    random_directions = random_directions ./ sqrt(sum(random_directions.^2,2));
    random_radii = rand(num_outliers,1) * max_radi;
    a = repmat(random_radii,1,dim_data);
    b = repmat(c_meb_whole,num_outliers,1);
    points = random_directions .* repmat(random_radii,1,dim_data) ...
        + repmat(c_meb_whole,num_outliers,1);
    
    generated_outliers = points;
    
    file_outlier = strcat(save_folder_outlier, '/', dataset_, '_outliers_', ...
        num2str(ratio_outliers), '.mat');
    save(file_outlier, 'generated_outliers');
    
    % verify
    dist_to_class = pdist2(generated_outliers, c_meb_class);
    dist_to_class_ratio = dist_to_class ./ repmat(r_meb_class',num_outliers,1);
    dist_to_c_meb = pdist2(generated_outliers, c_meb_whole);
    dist_to_c_meb_ratio = dist_to_c_meb / r_meb_whole;
    assert(sum(dist_to_c_meb_ratio>threshold_on_meb_whole) == 0);
    
    disp(['    inside MEB: ', num2str(sum(dist_to_c_meb <= r_meb_whole))]);
    disp(['    outside MEB: ', num2str(sum(dist_to_c_meb > r_meb_whole))]);
end

function [c_meb_class, r_meb_class] = get_meb_class(data, k, data_folder, dataset_)
    label = load([data_folder '/', dataset_, '_labels.mat'], 'labels');
    label_min = min(label.labels);
    label_max = max(label.labels);
    
    dim_data = size(data, 2);
    c_meb_class = zeros(k, dim_data);
    r_meb_class = zeros(k, 1);
    for i = label_min:label_max
        condition = label.labels==i;
        curr_data = data(condition, :);
        disp(['Calculating MEB of class ', num2str(i), '...']);
        if label_min == 0
            [c_meb_class(i+1,:), r_meb_class(i+1)] = alg_meb(curr_data, 0.01);
        else
            [c_meb_class(i,:), r_meb_class(i)] = alg_meb(curr_data, 0.01);
        end
    end
end
