function out_paths = assembleNNCtrl(ctrl_paths, template_path, output_dir, export_version)
% assembleNNCtrl puts trained NN controllers into a plant model.
%
% For each controller model (*_ctr_*.slx), the template plant model is copied,
% its Feed-Forward Neural Network block is replaced with the controller's, and
% the result is saved to output_dir under the controller's name without
% '_ctr' (e.g. X_ctr_Y.slx -> X_Y.slx). Works for AFC, ACC and WT: where the
% block sits is decided by the template name.
%
% Inputs:
%   ctrl_paths: controller model path, or a cell array of them
%   template_path: the plant model with the original NN controller
%   output_dir: where the assembled models are saved
%   export_version: Simulink version to save as (default 'R2021a')
% Outputs:
%   out_paths: paths of the assembled models
%
% Example (the AFC T=120 controllers 12 and 13):
%   b = 'benchmark/AFC/AFC_mu_015_spec_1_FFNN_trainlm_30_30_30_Nor_Jan_6';
%   m = fullfile(b, 'repaired_rq7', 'T_120', 'model');
%   c = fullfile(m, 'config&controller', 'AFC_FFNN_trainlm_30_30_30_st_5_ctr_Nor_Ep_1000_T_120_Sep_12_');
%   assembleNNCtrl({[c '12.slx'], [c '13.slx']}, ...
%       fullfile(b, 'model', 'AFC_FFNN_trainlm_30_30_30_Nor_Jan_6.slx'), m);

if nargin < 4
    export_version = 'R2021a';
end
ctrl_paths = cellstr(ctrl_paths);

[~, template_name] = fileparts(template_path);
if startsWith(template_name, 'AFC')
    nn_parent = '/Air Fuel Control Model 1';
elseif startsWith(template_name, {'ACC', 'WT'})
    nn_parent = '';
else
    error('assembleNNCtrl:unknownModel', 'No known NN block location for %s', template_name);
end
nn_block = 'Feed-Forward Neural Network';

if ~isfolder(output_dir)
    mkdir(output_dir);
end

out_paths = cell(size(ctrl_paths));
for k = 1:numel(ctrl_paths)
    [~, ctrl_base] = fileparts(ctrl_paths{k});
    if ~contains(ctrl_base, '_ctr_')
        error('assembleNNCtrl:name', 'Controller name must contain ''_ctr_'': %s', ctrl_base);
    end
    final_path = fullfile(output_dir, [strrep(ctrl_base, '_ctr_', '_') '.slx']);
    % Never overwrite: assembled models are committed, and regenerating one
    % silently would replace it with a different file.
    if isfile(final_path)
        error('assembleNNCtrl:exists', 'Already exists, remove it first to regenerate: %s', final_path);
    end

    tmp_name = sprintf('tmp_assembleNNCtrl_%d', k);
    tmp_path = fullfile(output_dir, [tmp_name '.slx']);
    copyfile(template_path, tmp_path);
    load_system(ctrl_paths{k});
    load_system(tmp_path);

    replaced = replace_block([tmp_name nn_parent], 'Name', nn_block, ...
        [ctrl_base '/' nn_block], 'noprompt');
    if isempty(replaced)
        error('assembleNNCtrl:notReplaced', 'NN block was not replaced in %s using %s', template_name, ctrl_base);
    end

    close_system(ctrl_base, 0);
    save_system(tmp_name);
    % ExportToVersion must save under a different name than the working copy.
    save_system(tmp_name, final_path, 'ExportToVersion', export_version);
    close_system(tmp_name, 0);
    [~, final_name] = fileparts(final_path);
    if bdIsLoaded(final_name)
        close_system(final_name, 0);
    end

    % Drop the working copy and the backup Simulink leaves when it upgrades it.
    delete(tmp_path);
    for f = dir(fullfile(output_dir, [tmp_name '.slx.r*']))'
        delete(fullfile(f.folder, f.name));
    end

    out_paths{k} = final_path;
    fprintf('[%d/%d] %s (%s)\n', k, numel(ctrl_paths), final_path, export_version);
end
end
