function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

H = find(y==1);
D = find(y==2);
A = find(y==3);

plot(X(H, 1), X(H, 2), 'b+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(D, 1), X(D, 2), 'yo', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
plot(X(A, 1), X(A, 2), 'kx', 'LineWidth', 2, 'MarkerSize', 7);


% =========================================================================



hold off;

end
