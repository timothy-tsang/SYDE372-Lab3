function [indice] = classifyMicdWithDistances(covariance,mean,feature1,feature2)
    listOfDistances = [];
    point = [feature1; feature2];
    
    % For each mean/covariance 
    for i=1:10
        % Equation for each point with all covariances/means 
        distance =((point-mean(:,i))'*inv(covariance{i})*(point-mean(:,i))).^0.5;
        
        % Add to array of distances 
        listOfDistances = [listOfDistances, distance];
    end
    
    % Return class with the minimal distance 
    [smallestElement, indice] = min(listOfDistances);
end