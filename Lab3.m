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


%% Section 3 - Labelled Classification 

% Find MICD Boundaries for f2, f8, f32:
% For f2: 
numImages = 10;
all_MICD_Vals = zeros(10,1);

%Reshape

M= [];
Test_Problems = zeros(10,16);
% Note: Each image has 16 blocks, where matrix is [i; j; image#; block (1-16)] 

% Data Set f2: 
% For each image, calculate mean for each point that corresponds to the image:
for i=0:numImages-1
    % Need Mean and Covariance for MICD
    % f2(1:2, 16(i)+1, 16(i+1)) - Get 1st and 2nd row, columns of every 16th
    set1 = f2(1:2,(16*i)+1:16*(i+1)); % Get i, j values for each image 
    Cov_1  = cov(set1');      % Calculate Covariance of set1
    M1 = mean(set1,2);        % Calculate mean of set1
        
    % compare against all classes
    for j = 0:size(set1,2)-1
        temp_matrix = find(f2(4,:) == j+1);
        for k = 1:length(temp_matrix) 
            temp_x = f2t(1:2,temp_matrix(k));
            all_MICD_Vals(k) = ged(Cov_1,M1',temp_x(1),temp_x(2));
        end  
    [~,index] = min(all_MICD_Vals);
    Test_Problems(i+1,j+1) = index;
    all_MICD_Vals = zeros(10,1);
    end    
end


% Create Mesh Grid for data set f2






% For f8:  TODO




% For f32:  TODO






%% Section 4 - Image Classification and Segmentation 






%% Section 5 - Unlabelled Clustering 
% Setup (For f32 only) 
data = f32;
features = f32(1:2,:)';
K = 10;

% TODO: Chenlei add your Part 5 here (make it a function) 
% K-means where K = 10 


% Fuzzy K-Means where b = 2
b = 2;
[cluster_centers, partition] = fcm(features, K, b);

% Find linespace of partition
maxPartition = max(partition);
index1 = find(partition(1,:) == maxPartition);
index2 = find(partition(2,:) == maxPartition);

% Plot Fuzzy K-Means 
figure;
title('Fuzzy K-Means where b = 2');
aplot(f32);
hold on;
scatter(cluster_centers(:,1), cluster_centers(:,2)); % Element U(i,j) indicates the degree of membership of the jth data point in the ith cluster