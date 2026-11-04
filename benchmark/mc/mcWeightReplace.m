function [mcNet] = mcWeightReplace(oriYamlFile, newYamlFile, altWeights, spsWeights)
% MCWEIGHTREPLACE Replaces specific weights in an MC network YAML file.
%
% Inputs
%   oriYamlFile: path to the original YAML file of nn controller
%   newYamlFile: path to save the updated YAML file
%   spsWeights: [li, i, j, val]
%
% Output
%   mcNet: struct representing the updated MC network

spsWNum = size(spsWeights,1);
tempSpsWeights = num2cell(spsWeights);

% layer field mapping
layerFields = {'x1', 'x2', 'x3'};

% load original nn controller in YAML
if ~endsWith(oriYamlFile, '.yml')
    oriYamlFile = [oriYamlFile, '.yml'];
end


oriYamlPath = which(oriYamlFile);
if isempty(oriYamlPath)
    error('File not found: %s', oriYamlFile);
else
    mcNet = yamlread(oriYamlPath);
end

for wi = 1:spsWNum
    [li,i,j] = tempSpsWeights{wi,1:3}; 
    mcNet.weights.(layerFields{li})(i,j) = altWeights(wi);
end

% save modified nn controller to YAML
if ~endsWith(newYamlFile, '.yml')
    newYamlFile = [newYamlFile, '.yml'];
end
yamlwrite(mcNet, newYamlFile);
end