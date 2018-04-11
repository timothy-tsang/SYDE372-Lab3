% Lab 3 
% Image Classification
% SYDE 372 - Lab 3
% Ali Akram 20526098
% Presish Bhattachan 20553154
% Chenlei Shen 20457272 
% Timothy Tsang 20556306 

%% Section 2 - Feature Analysis 
% Load Images and Data 
load feat.mat

% Setup for mean/covariance 
f8points = [];
f8Mean = [];
f8Covariance = [];
for i=1:10
    % Get points for each image 
    for j=1:16
        f8points(:,j) = f8(1:2,(i-1)*16+j);
    end
    f8Mean(:,i) = mean(f8points, 2); % Mean
    f8Covariance{i} = cov(f8points'); % Covariance
    clear f8points; % Reset 
end
%% Section 3 - Labelled Classification 

% Go through f2, f32, and f8 
numImages = 10;
all_MICD_Vals_f2 = zeros(10,1);
all_MICD_Vals_f32 = zeros(10,1);    
all_MICD_Vals_f8 = zeros(10,1);
labelled_Test_Data_f2 = zeros(10,16);
labelled_Test_Data_f32 = zeros(10,16);
labelled_Test_Data_f8 = zeros(10,16);

%Reshape
% Note: Each image has 16 blocks, where matrix is [i; j; image#; block (1-16)]
% Data Set f2
% Need Mean and Covariance for MICD
% For each image, calculate mean for each point that corresponds to the image:
for i=0:numImages-1
    setf2 = f2(1:2,(16*i)+1:16*(i+1)); % Get i, j values for each image 
    Cov_f2  = cov(setf2');      % Calculate Covariance of set1
    M_f2 = mean(setf2,2);        % Calculate mean of set1
    
    setf32 = f32(1:2,(16*i)+1:16*(i+1)); % Get i, j values for each image 
    Cov_f32  = cov(setf32');      % Calculate Covariance of set1
    M_f32 = mean(setf32,2);        % Calculate mean of set1
    
    setf8 = f8(1:2,(16*i)+1:16*(i+1)); % Get i, j values for each image 
    Cov_f8  = cov(setf8');      % Calculate Covariance of set1
    M_f8 = mean(setf8,2);        % Calculate mean of set1
    
    % compare against all classes
    for j = 0:size(setf2,2)-1
        temp_matrix = find(f2(4,:) == j+1);
        for k = 1:length(temp_matrix) 
            temp_x_f2 = f2t(1:2,temp_matrix(k));
            temp_x_f32 = f32t(1:2,temp_matrix(k));
            temp_x_f8 = f8t(1:2,temp_matrix(k));
            all_MICD_Vals_f2(k) = ged(Cov_f2,M_f2',temp_x_f2(1),temp_x_f2(2));
            all_MICD_Vals_f32(k) = ged(Cov_f32,M_f32',temp_x_f32(1),temp_x_f32(2));
            all_MICD_Vals_f8(k) = ged(Cov_f8,M_f8',temp_x_f8(1),temp_x_f8(2));
        end  
    [~,index2] = min(all_MICD_Vals_f2);
    [~,index32] = min(all_MICD_Vals_f32);
    [~,index8] = min(all_MICD_Vals_f8);
    labelled_Test_Data_f2(i+1,j+1) = index2;
    labelled_Test_Data_f32(i+1,j+1) = index32;
    labelled_Test_Data_f8(i+1,j+1) = index8;
    
    % Clear MICD 
    all_MICD_Vals_f2 = zeros(10,1);
    all_MICD_Vals_f32 = zeros(10,1);
    all_MICD_Vals_f8 = zeros(10,1);

    % clear_MICD();
    end 
end


%% Section 4 - Image Classification and Segmentation 
maxRows = 256; 
cimage = [];
for i = 1:maxRows
    for j = 1:maxRows
        feature1 = multf8(i,j,1);
        feature2 = multf8(i,j,2);
        cimage(i,j) = classifyMicdWithDistances(f8Covariance, f8Mean, feature1, feature2);
    end
end

% Plot image that was classified 
figure();
imagesc(cimage);
hold on;
title('Classified MultiImage');
xlabel('Feature 1')
ylabel('Feature 2')

% Plot multeImage multim
figure();
imagesc(multim);
colormap(gray);
hold on;
title('Original MultiImage');
xlabel('Feature 1')
ylabel('Feature 2')


%% Section 5 - Unlabelled Clustering 
% Setup (For f32 only)
data = f32;
features = data(1:2,:)';

% k clusters
K = 10;

% K-means where K = 10
% n samples
n = size(data, 2);
prototypes = zeros(2, k);
features = data(1:2,:);
range = randperm(n);
rand_posn = range(1:k);

for i=1:k
    prototypes(:, i) = features(:, rand_posn(i));
end

figure;
aplot(f32);
hold on
scatter(prototypes(1,:), prototypes(2,:));

% k-means algorithm
err_tol = 0.00001;
all_below_tol = false;

loop = 0;

while ~all_below_tol
    % get cluster k for each sample point
    samples_cluster = zeros(1, n);

    for i=1:n
        pt = features(:, i);
        min = Inf;
        min_dist = Inf;

        for j=1:k
            z = prototypes(:, j);
            dist = get_dist(pt, z);
            if dist < min_dist
                min = j;
                min_dist = dist;
            end
            samples_cluster(i) = min;
        end
    end

    % calculate new prototype using cluster mean
    new_prototypes = zeros(2, k);
    ten = [];

    for j=1:k
        cluster_pts = [];
        for i=1:n
            if samples_cluster(i) == j
                cluster_pts = [cluster_pts, features(:, i)];
            end
        end
        new_prototypes(:, j) = mean(cluster_pts, 2);
    end

    % compare new points with old points
    at_least_one_above_tol = false;

    for j=1:k
        if (prototypes(1,j) - new_prototypes(1,j) > err_tol...
            && prototypes(2,j) - new_prototypes(2,j) > err_tol)
            prototypes(:,j) = new_prototypes(:,j);
            at_least_one_above_tol = true;
        end
    end

    if ~at_least_one_above_tol
        all_below_tol = true;
    end

    loop = loop + 1;
end
scatter(prototypes(1,:), prototypes(2,:), 'g');
hold on;
title('K-Means where K = 10');
xlabel('Feature 1');
ylabel('Feature 2');

% Fuzzy K-Means where b = 2
b = 2;
[cluster_centers, partition] = fcm(features, K, b);

% Find linespace of partition
maxPartition = max(partition);
index1 = find(partition(1,:) == maxPartition);
index2 = find(partition(2,:) == maxPartition);

% Plot Fuzzy K-Means
figure;
aplot(f32);
hold on;
scatter(cluster_centers(:,1), cluster_centers(:,2)); % Element U(i,j) indicates the degree of membership of the jth data point in the ith cluster
hold on;
title('Fuzzy K-Means where b = 2');
xlabel('Feature 1');
ylabel('Feature 2');


%% Helper Functions 
function clear_MICD()
    all_MICD_Vals_f2 = zeros(10,1);
    all_MICD_Vals_f32 = zeros(10,1);
    all_MICD_Vals_f8 = zeros(10,1);
end

