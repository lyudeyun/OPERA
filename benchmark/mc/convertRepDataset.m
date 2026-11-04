% Batch convert Rep dataset
target_dir = fullfile(fileparts(mfilename('fullpath')), 'MC_Phi_1_FFNN_16_16_XV_100_100_Feb_11_2026', 'repaired', 'dataset');
files = dir(fullfile(target_dir, '*_Re.mat'));

fprintf('Found %d files to process.\n', length(files));

for k = 1:length(files)
    filename = files(k).name;
    full_path = fullfile(target_dir, filename);
    fprintf('Processing %s...\n', filename);
    
    data = load(full_path);
    if ~isfield(data, 'testSuite')
        warning('File %s does not contain testSuite. Skipping.', filename);
        continue;
    end
    
    testSuite = data.testSuite;
    num_traces = length(testSuite.testCases);
    
    % Determine T from the first test case
    if num_traces > 0 && ~isempty(testSuite.testCases(1).nnInputs)
        T = size(testSuite.testCases(1).nnInputs, 2);
    else
        warning('testSuite is empty or nnInputs is missing in %s.', filename);
        continue;
    end
    
    % Get dimensions
    dim_in = size(testSuite.testCases(1).nnInputs, 1);
    dim_out = size(testSuite.testCases(1).nnOutputs, 1);
    
    % Initialize cells with pre-allocation
    tr_ic_cell = cell(1, T);
    tr_oc_cell = cell(1, T);
    
    for t = 1:T
        tr_ic_cell{1, t} = zeros(dim_in, num_traces);
        tr_oc_cell{1, t} = zeros(dim_out, num_traces);
    end
    
    fprintf('  Extracting data (T=%d, N=%d)...\n', T, num_traces);
    for i = 1:num_traces
        nnInputs = testSuite.testCases(i).nnInputs;
        nnOutputs = testSuite.testCases(i).nnOutputs;
        
        if ~isempty(nnInputs) && ~isempty(nnOutputs)
            for j = 1:T
                tr_ic_cell{1,j}(:, i) = nnInputs(:,j);
                tr_oc_cell{1,j}(:, i) = nnOutputs(:,j);
            end
        end
    end
    
    % Construct new filename
    [~, name, ext] = fileparts(filename);
    new_filename = [name, '_New', ext];
    save_path = fullfile(target_dir, new_filename);
    
    % Save
    save(save_path, 'tr_ic_cell', 'tr_oc_cell');
    fprintf('  Saved to %s\n', new_filename);
end

fprintf('All done.\n');
