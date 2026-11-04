function samples = gridSampling2D(xMin, xMax, yMin, yMax, numGridX, numGridY)
% GRIDSAMPLING2D Samples one random point in each cell of a 2D grid.
%   
% Inputs:
%   xMin, xMax - Range of x
%   yMin, yMax - Range of y
%   numGridX   - Number of grid cells along the x-axis
%   numGridY   - Number of grid cells along the y-axis
%
% Output:
%   samples    - A matrix of size (numGridX * numGridY) × 2 where each row is a sampled [x, y] point

xEdges = linspace(xMin, xMax, numGridX + 1);
yEdges = linspace(yMin, yMax, numGridY + 1);

samples = zeros(numGridX * numGridY, 2);
idx = 1;
for i = 1:numGridX
    for j = 1:numGridY
        xStart = xEdges(i);
        xEnd = xEdges(i+1);
        yStart = yEdges(j);
        yEnd = yEdges(j+1);
        samples(idx, :) = [ ...
            xStart + (xEnd - xStart) * rand(), ...
            yStart + (yEnd - yStart) * rand() ...
            ];
        idx = idx + 1;
    end
end
end