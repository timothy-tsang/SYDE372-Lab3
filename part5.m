clf;
image1 = readim('cloth.im');
image2 = readim('cork.im');

subplot( 2, 1, 1 )
imagesc(1, 1, image1);
subplot( 2, 1, 2 )
imagesc(1, 2, image2);
colormap(gray) ;

%% getting f32
load feat.mat
data = f32;

%% generating random prototpyes
% k clusters
k = 10;
% n samples
n = size(f32, 2);
prototypes = zeros(2, k);
features = f32(1:2,:);
range = randperm(n);
rand_posn = range(1:k);

for i=1:k
    prototypes(:, i) = features(:, rand_posn(i));
end

figure;
scatter(features(1,:), features(2,:));
hold on
scatter(prototypes(1,:), prototypes(2,:));

%% k-means algorithm
err_tol = 0.00001;
all_below_tol = false;

loop = 0;

while ~all_below_tol
    %% get cluster k for each sample point
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

    %% calculate new prototype using cluster mean
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

    %scatter(new_prototypes(1,:), new_prototypes(2,:));

    %% compare new points with old points
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


scatter(prototypes(1,:), prototypes(2,:));