function net = netFromModel(mdl_path, train_fcn, out_path)
% netFromModel rebuilds the network object of a model's NN controller.
%
% Reads the weights, biases and transfer functions from the gensim-generated
% "Feed-Forward Neural Network" block of the model and builds the same
% feedforwardnet (no input/output processing, as in the trained controllers).
% Used for controllers whose network file is missing.
%
% A network object saved by a newer MATLAB cannot be loaded by an older one
% (it is read in as uint32), so save it with the oldest MATLAB that will use it.
%
% Inputs:
%   mdl_path: path to the model (.slx)
%   train_fcn: training function the controller was trained with, e.g. 'trainlm'
%   out_path: optional; when given, net is saved there (with a note on its source)
% Outputs:
%   net: the rebuilt network
%
% Example (AFC#1):
%   b = 'benchmark/AFC/AFC_mu_015_spec_1_FFNN_trainlm_15_15_15_NoSat_Jan_23/model';
%   netFromModel(fullfile(b, 'AFC_FFNN_trainlm_15_15_15_NoSat_Jan_23.slx'), 'trainlm', ...
%       fullfile(b, 'config&controller', 'AFC_FFNN_trainlm_15_15_15_config_NoSat_Jan_23.mat'));

[~, mdl] = fileparts(mdl_path);
was_loaded = bdIsLoaded(mdl);
load_system(mdl_path);

nnb = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Name', 'Feed-Forward Neural Network');
if numel(nnb) ~= 1
    error('netFromModel:block', 'Expected one Feed-Forward Neural Network block in %s, found %d', mdl, numel(nnb));
end
nnb = nnb{1};
n_layer = numel(find_system(nnb, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'SearchDepth', 1, ...
    'Regexp', 'on', 'Name', '^Layer \d+$'));
tf_names = {'poslin', 'purelin', 'tansig', 'logsig', 'satlin', 'satlins', 'hardlim', 'radbas', 'softmax', 'elliotsig'};

W = cell(1, n_layer);
b = cell(1, n_layer);
tf = cell(1, n_layer);
for k = 1:n_layer
    layer = sprintf('%s/Layer %d', nnb, k);
    b{k} = reshape(str2num(get_param(sprintf('%s/b{%d}', layer, k), 'Value')), [], 1); %#ok<ST2NM>
    kids = get_param(find_system(layer, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'SearchDepth', 1), 'Name');
    tf{k} = kids{ismember(kids, tf_names)};
    % gensim names the rows of every layer's weight block IW{k,k-1}(i,:)'
    if k == 1
        w_block = [layer '/IW{1,1}'];
        row_fmt = 'IW{1,1}(%d,:)''';
    else
        w_block = sprintf('%s/LW{%d,%d}', layer, k, k-1);
        row_fmt = sprintf('IW{%d,%d}(%%d,:)''', k, k-1);
    end
    rows = cell(numel(b{k}), 1);
    for i = 1:numel(b{k})
        rows{i} = reshape(str2num(get_param([w_block '/' sprintf(row_fmt, i)], 'Value')), 1, []); %#ok<ST2NM>
    end
    W{k} = vertcat(rows{:});
end
if ~was_loaded
    close_system(mdl, 0);
end

n_in = size(W{1}, 2);
net = feedforwardnet(cellfun(@numel, b(1:end-1)), train_fcn);
net.inputs{1}.processFcns = {};
net.outputs{n_layer}.processFcns = {};
net = configure(net, [zeros(n_in, 1) ones(n_in, 1)], [0 1]);
net.name = 'Feed-Forward Neural Network';
for k = 1:n_layer
    net.layers{k}.transferFcn = tf{k};
    net.b{k} = b{k};
end
net.IW{1,1} = W{1};
for k = 2:n_layer
    net.LW{k,k-1} = W{k};
end

if nargin >= 3
    source = sprintf('Rebuilt on %s with MATLAB %s from the Feed-Forward Neural Network block of %s.slx.', ...
        datestr(now, 'yyyy-mm-dd'), version('-release'), mdl); %#ok<NASGU>
    save(out_path, 'net', 'source');
    fprintf('Saved %s (layers %s)\n', out_path, mat2str(cellfun(@(l) l.size, net.layers)'));
end
end
