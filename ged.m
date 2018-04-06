function distance = ged(cov1, mean1, x,y) 
   % Transposing the covariances 
   inv1 = inv(cov1);
   xBar = [x, y];
   distance = ((xBar - mean1) * inv1 * (xBar - mean1)');
     % end 
end