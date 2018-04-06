function dist = getDist( point, prototype )
    dist = sqrt((point(1)-prototype(1))^2 + (point(2)-prototype(2))^2);
end