function out_path = makeRepFastModel(rep_path, out_dir)
% makeRepFastModel builds a fast-restart friendly copy of a *_Rep model.
%
% The *_Rep models read patch_sig, safe_patch_sig and sw through DSP "Signal
% From Workspace" blocks, whose data is fixed when the model compiles: in fast
% restart every simulation would keep the first patch. Each such block is
% replaced by a counter driving a direct lookup table over the same variable.
% The counter counts samples of the block's own sample time, so sample k
% outputs element k+1 of the variable, as the DSP block did, and the table is a
% tunable parameter, so it is read again at the start of every simulation.
%
% The original model is not modified; the copy is saved as <name>Fast.slx
% (e.g. X_Rep.slx -> X_RepFast.slx) in out_dir, in the same Simulink release
% as the original so that machines with the original's MATLAB can open it.
%
% Inputs:
%   rep_path: path of the *_Rep model
%   out_dir: folder for the new model
% Outputs:
%   out_path: path of the new model

[~, name] = fileparts(rep_path);
new_name = [name 'Fast'];
out_path = fullfile(out_dir, [new_name '.slx']);
if isfile(out_path)
    error('makeRepFastModel:exists', 'Already exists: %s', out_path);
end

release = Simulink.MDLInfo(rep_path).ReleaseName;
% Edit a working copy and export it at the end: ExportToVersion cannot save
% a model under its own name.
work_name = [new_name '_work'];
work_path = fullfile(tempdir, [work_name '.slx']);
load_system(rep_path);
save_system(name, work_path);   % the loaded model is now work_name; rep_path is untouched

dsp_src = sprintf('dspsrcs4/Signal From\nWorkspace');
blks = find_system(work_name, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'ReferenceBlock', dsp_src);
if isempty(blks)
    blks = find_system(work_name, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'SourceBlock', dsp_src);
end
if isempty(blks)
    close_system(work_name, 0);
    delete(work_path);
    error('makeRepFastModel:noDSP', 'No DSP Signal From Workspace block in %s', name);
end

for k = 1:numel(blks)
    blk = blks{k};
    var_name = get_param(blk, 'X');
    ts = get_param(blk, 'Ts');
    if ~strcmp(get_param(blk, 'nsamps'), '1') || ~strcmp(get_param(blk, 'OutputFrames'), 'off')
        error('makeRepFastModel:unsupported', 'Only one sample per step is supported: %s', blk);
    end
    parent = get_param(blk, 'Parent');
    blk_name = get_param(blk, 'Name');
    pos = get_param(blk, 'Position');
    ph = get_param(blk, 'PortHandles');
    ln = get_param(ph.Outport, 'Line');
    dst = get_param(ln, 'DstPortHandle');
    delete_line(ln);
    delete_block(blk);

    new_blk = [parent '/' blk_name];
    add_block('built-in/Subsystem', new_blk, 'Position', pos);
    add_block('simulink/Sources/Counter Limited', [new_blk '/Sample index'], ...
        'uplimit', sprintf('numel(%s)-1', var_name), 'tsamp', ts, 'Position', [30 40 80 70]);
    add_block('simulink/Lookup Tables/Direct Lookup Table (n-D)', [new_blk '/Sample value'], ...
        'NumberOfTableDimensions', '1', 'Table', var_name, ...
        'DiagnosticForOutOfRangeInput', 'None', 'Position', [130 35 190 75]);
    add_block('built-in/Outport', [new_blk '/Out1'], 'Position', [240 48 270 62]);
    add_line(new_blk, 'Sample index/1', 'Sample value/1');
    add_line(new_blk, 'Sample value/1', 'Out1/1');

    new_ph = get_param(new_blk, 'PortHandles');
    for j = 1:numel(dst)
        add_line(parent, new_ph.Outport, dst(j));
    end
    fprintf('replaced %s (X = %s, Ts = %s)\n', strrep(new_blk, newline, ' '), var_name, ts);
end

save_system(work_name);
save_system(work_name, out_path, 'ExportToVersion', release);
close_system(work_name, 0);
if bdIsLoaded(new_name)
    close_system(new_name, 0);
end
delete(work_path);
for f = dir(fullfile(out_dir, [new_name '.slx.r*']))'   % backup Simulink leaves when exporting
    delete(fullfile(f.folder, f.name));
end
end
