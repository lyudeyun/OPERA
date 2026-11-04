function generateDataSetParallel(mdl, D_size, D_name, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, oc_name, phi, num_workers)
% generateDataSetParallel generates traces with independent worker models.
%
% Each worker:
%   1. copies the source model using a worker-numbered model name;
%   2. initializes its own BreachSimulinkSystem;
%   3. simulates an independent subset of traces.
%
% Results are merged in the original trace-index order and saved using the
% same variables and layout as generateDataSet.

if nargin < 14
    num_workers = 4;
end
if D_size < 1 || fix(D_size) ~= D_size
    error('D_size must be a positive integer.');
end
if ~isempty(num_workers) && num_workers <= 1
    generateDataSet( ...
        mdl, D_size, D_name, T, Ts, in_name, in_range, in_span, ...
        icc_name, ic_const, ics_name, oc_name, phi);
    return;
end

pool = gcp('nocreate');
if isempty(pool)
    if isempty(num_workers)
        parpool('local');
    else
        parpool('local', num_workers);
    end
elseif ~isempty(num_workers) && pool.NumWorkers ~= num_workers
    warning( ...
        'generateDataSetParallel:ExistingPoolSize', ...
        ['Using the existing pool with %d workers instead of the requested ' ...
         '%d workers.'], ...
        pool.NumWorkers, num_workers);
end

[model_path, workspace_variables] = getWorkerModelContext(mdl);
run_id = randi(99999999);
run_folder = fullfile( ...
    tempdir, sprintf('opera_gendata_%08d', run_id));
while isfolder(run_folder)
    run_id = randi(99999999);
    run_folder = fullfile( ...
        tempdir, sprintf('opera_gendata_%08d', run_id));
end
mkdir(run_folder);

worker_system = parallel.pool.Constant( ...
    @() initializeWorkerSystem( ...
        model_path, T, Ts, workspace_variables, run_id, run_folder), ...
    @cleanupWorkerSystem);
run_cleanup = onCleanup(@() cleanupParallelRun( ...
    worker_system, run_folder));

tr_in_cell = cell(1, D_size);
trace_ic_cell = cell(1, D_size);
trace_oc_cell = cell(1, D_size);
tr_rob_set = zeros(D_size, 1);
parallel_seed = randi(intmax('int32'));

parfor tr_idx = 1:D_size
    stream = RandStream('mrg32k3a', 'Seed', parallel_seed);
    stream.Substream = tr_idx;
    RandStream.setGlobalStream(stream);

    worker = worker_system.Value;
    fprintf('worker %d: trace %d\n', worker.Number, tr_idx);
    [tr_in_cell{1, tr_idx}, trace_ic_cell{1, tr_idx}, ...
        trace_oc_cell{1, tr_idx}, tr_rob_set(tr_idx, 1)] = oneSimParallel( ...
        worker.Br, T, Ts, in_name, in_range, in_span, ...
        icc_name, ic_const, ics_name, oc_name, phi);
end

tr_ic_cell = trace_ic_cell{1};
tr_oc_cell = trace_oc_cell{1};
for tr_idx = 2:D_size
    tr_ic_cell = catsamples( ...
        tr_ic_cell, trace_ic_cell{tr_idx}, 'pad');
    tr_oc_cell = catsamples( ...
        tr_oc_cell, trace_oc_cell{tr_idx}, 'pad');
end

save(D_name, 'tr_in_cell', 'tr_ic_cell', 'tr_oc_cell', 'tr_rob_set');
clear run_cleanup;
end


function [model_path, workspace_variables] = getWorkerModelContext(mdl)
was_loaded = bdIsLoaded(mdl);
if ~was_loaded
    load_system(mdl);
end

model_path = which([mdl, '.slx']);
if isempty(model_path)
    model_path = get_param(mdl, 'FileName');
end
if isempty(model_path)
    error('Unable to locate the Simulink model file for: %s', mdl);
end

variable_usage = Simulink.findVars( ...
    mdl, 'SearchMethod', 'compiled', ...
    'SourceType', 'base workspace');
variable_names = unique({variable_usage.Name}, 'stable');
workspace_variables = struct();
for var_idx = 1:numel(variable_names)
    var_name = variable_names{var_idx};
    workspace_variables.(var_name) = evalin('base', var_name);
end

if ~was_loaded
    close_system(mdl, 0);
end
end


function worker = initializeWorkerSystem( ...
        model_path, T, Ts, workspace_variables, run_id, run_folder)
task = getCurrentTask();
if isempty(task)
    worker.Number = 1;
else
    worker.Number = task.ID;
end

worker.PreviousFolder = pwd;
worker.WorkFolder = fullfile( ...
    run_folder, sprintf('worker_%d', worker.Number));
if ~isfolder(worker.WorkFolder)
    mkdir(worker.WorkFolder);
end
cd(worker.WorkFolder);

worker.ModelName = sprintf( ...
    'ar_%08d_w%d', run_id, worker.Number);
worker.ModelPath = fullfile( ...
    worker.WorkFolder, [worker.ModelName, '.slx']);
copyfile(model_path, worker.ModelPath);

variable_names = fieldnames(workspace_variables);
for var_idx = 1:numel(variable_names)
    var_name = variable_names{var_idx};
    assignin('base', var_name, workspace_variables.(var_name));
end

addpath(fileparts(model_path));
InitBreach;
breach_init_path = which('InitBreach');
breach_root = fileparts(breach_init_path);
worker.BreachModelPrefix = fullfile( ...
    breach_root, 'Ext', 'ModelsData', ...
    [worker.ModelName, '_breach']);

load_system(worker.ModelPath);
worker.Br = BreachSimulinkSystem(worker.ModelName);
worker.Br.Sys.tspan = 0:Ts:T;
fprintf( ...
    'Initialized worker %d with model %s\n', ...
    worker.Number, worker.ModelName);
end


function cleanupWorkerSystem(worker)
try
    bdclose('all');
catch
end
try
    cd(worker.PreviousFolder);
catch
    cd(tempdir);
end

breach_files = dir([worker.BreachModelPrefix, '*']);
for file_idx = 1:numel(breach_files)
    if ~breach_files(file_idx).isdir
        try
            delete(fullfile( ...
                breach_files(file_idx).folder, ...
                breach_files(file_idx).name));
        catch
        end
    end
end

if isfolder(worker.WorkFolder)
    try
        rmdir(worker.WorkFolder, 's');
    catch
    end
end
end


function cleanupParallelRun(worker_system, run_folder)
try
    delete(worker_system);
catch
end
if isfolder(run_folder)
    try
        rmdir(run_folder, 's');
    catch
    end
end
end


function [in_cell, ic_cell, oc_cell, rob] = oneSimParallel( ...
        Br, T, Ts, in_name, in_range, in_span, ...
        icc_name, ic_const, ics_name, oc_name, phi)
in_span = cell2mat(in_span);
if isscalar(unique(in_span))
    con_type = 'UniStep';
    input_gen.type = con_type;
    input_gen.cp = T / in_span(1);
else
    con_type = 'VarStep';
    input_gen.type = con_type;
    input_gen.cp = T ./ in_span;
end

Br.SetInputGen(input_gen);

in_cell = {};
ic_mat = [];
oc_mat = [];

for i = 1:numel(in_name)
    if strcmp(con_type, 'UniStep')
        input_cp = input_gen.cp;
        input_values = unifrnd( ...
            in_range{1,i}(1,1), in_range{1,i}(1,2), ...
            1, input_cp);
    else
        input_cp = input_gen.cp(i);
        input_values = unifrnd( ...
            in_range{1,i}(1,1), in_range{1,i}(1,2), ...
            1, input_cp);
    end
    in_cell{1,i} = input_values;

    for cpi = 0:input_cp - 1
        Br.SetParam( ...
            {[in_name{1,i}, '_u', num2str(cpi)]}, ...
            input_values(1,cpi+1));
        if strcmp(con_type, 'VarStep') && cpi ~= input_cp - 1
            Br.SetParam( ...
                {[in_name{1,i}, '_dt', num2str(cpi)]}, ...
                in_span(1,i));
        end
    end
end

Br.Sim(0:Ts:T);

ic_sig_val = sigMatch(Br, ics_name);
oc_sig_val = sigMatch(Br, oc_name);

for i = 1:numel(icc_name)
    ic_mat = [ ...
        ic_mat; ...
        ic_const{1,i} * ones(1, T/Ts+1)]; %#ok<AGROW>
end

ic_mat = [ic_mat; ic_sig_val];
ic_cell = num2cell(ic_mat,1);
oc_mat = [oc_mat; oc_sig_val];
oc_cell = num2cell(oc_mat,1);

rob = Br.CheckSpec(phi);
disp(rob);
end
