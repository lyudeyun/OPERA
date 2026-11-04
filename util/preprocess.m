function nor_path = preprocess(dataset, benchmark)
% preprocess Normalize a repaired dataset (*_Re.mat) with mapminmax.
%
% Inputs:
%   dataset: path to *_Re.mat (default: AFC trainlm 30_30_30 St_5 Re)
%   benchmark: 'AFC' | 'ACC' | 'WT' (default: 'AFC')
% Outputs:
%   nor_path: path to the saved Nor_*_Re.mat
%
% Saves: tr_in_cell, tr_ic_cell, tr_oc_cell, tr_rob_set, ps_x, ps_y

if nargin < 1 || isempty(dataset)
	root = fileparts(fileparts(mfilename('fullpath')));
	dataset = fullfile(root, ...
		'benchmark/AFC/AFC_mu_015_spec_1_FFNN_trainlm_30_30_30_Nor_Jan_6/repaired/dataset', ...
		'AFC_mu_015_spec_1_FFNN_trainlm_30_30_30_Nor_Jan_6_Tr_St_5_Re.mat');
end
if nargin < 2 || isempty(benchmark)
	benchmark = 'AFC';
end

[dataset_dir, dataset_name, dataset_ext] = fileparts(dataset);
if isempty(dataset_ext)
	dataset_ext = '.mat';
end
if isempty(dataset_dir)
	dataset_dir = pwd;
end
dataset_path = fullfile(dataset_dir, [dataset_name, dataset_ext]);
nor_path = fullfile(dataset_dir, ['Nor_', dataset_name, dataset_ext]);

if exist(nor_path, 'file')
	fprintf('Normalized dataset already exists, skip: %s\n', nor_path);
	return;
end
if ~exist(dataset_path, 'file')
	error('preprocess:DatasetNotFound', 'Dataset not found: %s', dataset_path);
end

fprintf('Loading: %s\n', dataset_path);
S = load(dataset_path);
required = {'tr_in_cell', 'tr_ic_cell', 'tr_oc_cell', 'tr_rob_set'};
for k = 1:numel(required)
	if ~isfield(S, required{k})
		error('preprocess:MissingField', 'Missing %s in %s', required{k}, dataset_path);
	end
end

tr_in_cell = S.tr_in_cell;
tr_ic_cell = S.tr_ic_cell;
tr_oc_cell = S.tr_oc_cell;
tr_rob_set = S.tr_rob_set;
clear S;

% ACC: drop the first two ic channels
if strcmp(benchmark, 'ACC')
	for idx = 1:numel(tr_ic_cell)
		tr_ic_cell{1, idx} = tr_ic_cell{1, idx}(3:5, :);
	end
end

% dataIn, dataOut
dataIn = tr_ic_cell;
dataOut = tr_oc_cell;

% preprocess input data
dataIn_Mat = cell2mat(dataIn);
% mapminmax
[~, ps_x] = mapminmax(dataIn_Mat, 0, 1);
index_in = find((ps_x.xmax - ps_x.xmin) == 0);
ps_x.gain(index_in) = 1.0 ./ ps_x.xmax(index_in);
ps_x.xmax(index_in) = 1;
ps_x.xmin(index_in) = 1;
nor_dataIn = mapminmax('apply', dataIn, ps_x);

% preprocess output data
dataOut_Mat = cell2mat(dataOut);
% mapminmax
[~, ps_y] = mapminmax(dataOut_Mat, 0, 1);
index_out = find((ps_y.xmax - ps_y.xmin) == 0);
ps_y.gain(index_out) = 1.0 ./ ps_y.xmax(index_out);
ps_y.xmax(index_out) = 1;
ps_y.xmin(index_out) = 1;
nor_dataOut = mapminmax('apply', dataOut, ps_y);

tr_ic_cell = nor_dataIn;
tr_oc_cell = nor_dataOut;

fprintf('Saving: %s\n', nor_path);
save(nor_path, 'tr_rob_set', 'tr_in_cell', 'tr_ic_cell', 'tr_oc_cell', 'ps_x', 'ps_y', '-v7.3');
fprintf('Done. benchmark=%s\n', benchmark);
end
