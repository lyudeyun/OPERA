function [mcNet] = convertMCYamlToMat(yamlFile)
% CONVERTMCYAMLTOMAT Converts an MC neural network defined in a YAML file 
% into a MATLAB struct compatible with MATLAB's NN toolbox conventions.
%
% Input:
%   yamlFile - path to the YAML file containing the MC network definition
%
% Output:
%   mcNet - struct with fields:
%       .name       - (empty) optional network name
%       .numlayers  - number of layers in the network (set to 3)
%       .IW         - input weights (cell array)
%       .LW         - layer weights (cell array)
%       .b          - layer biases (cell array)
%       .layers     - layer definitions including activation functions

% load yaml file
tempNet = yamlread(yamlFile);

% Initialize net structure
mcNet = struct();
mcNet.name = 'Feed-Forward Neural Network';
mcNet.numLayers = 3;
mcNet.IW = cell(3,1);
mcNet.LW = cell(3,3);
mcNet.b = cell(3,1);
mcNet.layers = cell(3,1);

% activation function
mcNet.layers{1}.transferFcn = 'logsig';
mcNet.layers{2}.transferFcn = 'logsig';
mcNet.layers{3}.transferFcn = 'tansig';

% weight assignment
mcNet.IW{1,1} = tempNet.weights.x1;
mcNet.LW{2,1} = tempNet.weights.x2;
mcNet.LW{3,2} = tempNet.weights.x3;

% bias assignment
mcNet.b{1} = tempNet.offsets.x1';
mcNet.b{2} = tempNet.offsets.x2';
mcNet.b{3} = tempNet.offsets.x3';
end