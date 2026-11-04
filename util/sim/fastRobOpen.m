function h = fastRobOpen(mdlInfo, simConf)
% FASTROBOPEN  Open a per-worker simulation session, built once and reused.
%
% The session holds Breach's instrumented copy of the faulty model, loaded and
% kept in fast restart. Evaluating an individual then costs only a few block
% parameter writes instead of rebuilding a model and a BreachSimulinkSystem.
%
% A worker keeps one session alive for the whole run; it is rebuilt only if the
% model, horizon or suspicious-weight set changes.

persistent CTX KEY GUARD BUILDS

% Called with no arguments, suspend instead: leave the session cached but take
% the model out of fast restart, so MATLAB can exit without prompting. The
% search does this every time it finishes, which happens dozens of times per
% experiment -- tearing the session down each time would mean rebuilding the
% model and the Breach interface just as often.
if nargin == 0
    if ~isempty(CTX) && bdIsLoaded(CTX.mdl)
        set_param(CTX.mdl, 'FastRestart', 'off');
    end
    h = [];
    return;
end

% The session caches the spec and the input layout too, so both belong in the
% key -- otherwise a worker that switches specification would silently keep
% scoring against the previous one.
% Signals recorded besides those of the spec (simConf.extraSigs, optional), for
% callers that compute metrics on other outputs.
if isfield(simConf, 'extraSigs')
    extraSigs = reshape(cellstr(simConf.extraSigs), 1, []);
else
    extraSigs = {};
end
key = sprintf('%s|%.17g|%.17g|%s|%s|%s|%s|%s', mdlInfo.faultyMdl, simConf.T, simConf.Ts, ...
              mat2str(mdlInfo.spsWeights), disp(simConf.phi), ...
              mat2str(simConf.in_span), strjoin(simConf.in_name, ','), strjoin(extraSigs, ','));

if ~isempty(CTX) && strcmp(KEY, key)
    h = CTX;
    return;
end

if isempty(BUILDS), BUILDS = 0; end
BUILDS = BUILDS + 1;
h = buildSession(mdlInfo, simConf, extraSigs, BUILDS);
CTX = h;
KEY = key;

% A model left in fast restart makes MATLAB prompt before it can be closed,
% which fails a batch job at exit. Held only here, so the guard runs once when
% persistents are cleared rather than on every returned copy of h.
GUARD = onCleanup(@() releaseModel(h.mdl));
end


function releaseModel(mdl)
if bdIsLoaded(mdl)
    set_param(mdl, 'FastRestart', 'off');
    close_system(mdl, 0);
end
end


function h = buildSession(mdlInfo, simConf, extraSigs, buildNo)
h = struct();
h.T       = simConf.T;
h.Ts      = simConf.Ts;
h.tspan   = 0:simConf.Ts:simConf.T;
h.in_name = simConf.in_name;
h.phi     = simConf.phi;

in_span = simConf.in_span;
if isscalar(unique(in_span))
    h.cp = repmat(simConf.T/in_span(1), 1, numel(simConf.in_name));
else
    h.cp = simConf.T./in_span;
end

h.sigNames = reshape(cellstr(STL_ExtractSignals(simConf.phi)), 1, []);
h.sigNames = unique([h.sigNames, extraSigs], 'stable');

% One trace system for the session rather than one per test case: constructing
% it runs InitBreach, and at one per test case that is on the order of a
% million calls per experiment. It is a handle object, so the copies of h
% handed back to callers all share it.
h.trace = BreachTraceSystem(h.sigNames);

% Breach names its copy '<mdl>_breach' unconditionally, so anything sharing one
% Breach installation would fight over the same file. Each worker therefore
% takes its own copy, named after the process.
%
% The pid is what makes the name unique across concurrent *jobs*: several
% experiments started by hand on one workstation share a working directory and
% a ModelsData directory, and a worker numbered within its own pool collides
% with the same-numbered worker of another job. That is what took down a
% production run -- one job deleted and recreated the copy another was loading.
%
% The name is also kept short. A model name is a MATLAB identifier and cannot
% exceed namelengthmax; several benchmark models are long enough that carrying
% the name over and appending both this tag and Breach's '_breach' would exceed
% it. A readable prefix plus a digest keeps different models apart.
mdl = mdlInfo.faultyMdl;
digest = lower(strrep(mat2str(mod(sum(double(mdl) .* (1:numel(mdl))), 1e6)), '-', ''));
% Each build gets its own name. Reusing one across rebuilds means inheriting
% the previous build's instrumented model, Breach's cached interface for it and
% whatever state it was left in; building a fresh name keeps every rebuild
% structurally identical to a first build, which is the path that is exercised
% on every run.
localMdl = sprintf('fr_%s_%s_w%d_%d', mdl(1:min(end, 12)), digest, feature('getpid'), buildNo);
localPath = fullfile(pwd, [localMdl '.slx']);

% Weights are pushed into the loaded model, never saved, so an existing copy is
% still pristine and is reused rather than deleted -- deleting it would leave a
% window where a rebuild cannot find the model at all.
if ~isfile(localPath)
    copyfile(which([mdl '.slx']), localPath);
end
discardCopiesOfDeadWorkers(pwd);
discardCopiesOfDeadWorkers(BreachGetModelsDataPath());
stale = [localMdl '_breach'];
if bdIsLoaded(stale)
    set_param(stale, 'FastRestart', 'off');
    bdclose(stale);
end
if bdIsLoaded(localMdl), bdclose(localMdl); end

% Load it by full path. Breach resolves the model from its bare name, which
% depends on the current folder and the MATLAB path being what they were when
% the copy was made -- a production run failed exactly there. Loading it here
% means the bare-name lookup finds a model that is already in memory.
load_system(localPath);

% Drive Breach's instrumented copy rather than the original model: a separate
% compilation of the same model differs in the last bits, and the stiff
% benchmarks amplify that into robustness differences large enough to flip signs.
Br = BreachSimulinkSystem(localMdl);
Br.Sys.tspan = h.tspan;
h.mdl       = Br.Sys.mdl;
h.inputList = Br.Sys.InputList;

% Breach caches interfaces in a global map keyed on the model file, and on a
% hit it returns a copy without running CreateInterface -- so nothing has
% loaded the instrumented model. Load it from where Breach keeps it.
if ~bdIsLoaded(h.mdl)
    load_system(fullfile(BreachGetModelsDataPath(), [h.mdl '.slx']));
end

ports = find_system(h.mdl, 'SearchDepth', 1, 'BlockType', 'Outport');
h.outportNames = cell(1, numel(ports));
for i = 1:numel(ports)
    h.outportNames{str2double(get_param(ports{i}, 'Port'))} = ...
        strrep(regexprep(ports{i}, ['^' regexptranslate('escape', h.mdl) '/'], ''), ' ', '_');
end

% u__ columns follow Breach's InputList, which is Inport port order and not
% necessarily the order in_name is listed in.
h.portOfInput = nan(1, numel(simConf.in_name));
for i = 1:numel(simConf.in_name)
    k = find(strcmp(h.inputList, simConf.in_name{i}), 1);
    if isempty(k)
        error('fastRobOpen:inputUnmatched', '%s is not an input of %s', simConf.in_name{i}, h.mdl);
    end
    h.portOfInput(i) = k;
end

% Cache each suspicious row's Value string split into entries, so that setting
% weights can substitute only the suspicious columns and leave every other
% entry's original text alone -- exactly what replaceWeights does. Reformatting
% them would perturb the last bits, since %.16g does not round-trip a double.
h.spsWeights = mdlInfo.spsWeights;
h.rowBlocks  = unique(mdlInfo.spsWeights(:, 1:2), 'rows');
h.rowEntries = cell(1, size(h.rowBlocks, 1));
h.rowPaths   = cell(1, size(h.rowBlocks, 1));
for ri = 1:size(h.rowBlocks, 1)
    layerIdx = h.rowBlocks(ri, 1);
    rowIdx   = h.rowBlocks(ri, 2);
    h.rowPaths{ri} = weightBlockPath(h.mdl, mdlInfo.faultyMdl, layerIdx, rowIdx);
    str = get_param(h.rowPaths{ri}, 'Value');
    h.rowEntries{ri} = strsplit(erase(erase(str, '['), ']'), ';');

    % Same check replaceWeights made: the block must actually hold the weights
    % oriWeights says it does, which catches a spsWeights index pointing at the
    % wrong neuron or a mismatched network file.
    inModel = str2num(str); %#ok<ST2NM>
    for colIdx = mdlInfo.spsWeights(mdlInfo.spsWeights(:,1) == layerIdx & ...
                                    mdlInfo.spsWeights(:,2) == rowIdx, 3)'
        expected = mdlInfo.oriWeights{1, layerIdx}(rowIdx, colIdx);
        if abs(inModel(colIdx) - expected) > 1e-10
            error('fastRobOpen:weightMismatch', ...
                'Mismatch at [Layer %d, Row %d, Col %d]: model = %.16f, expected = %.16f', ...
                layerIdx, rowIdx, colIdx, inModel(colIdx), expected);
        end
    end
end

% Breach configures ExternalInput as the expression '[t__, u__]', so inputs go
% through the base workspace. The output grid must be pinned before fast restart
% is enabled, because sim() rejects OutputOption/OutputTimes once it is on.
set_param(h.mdl, 'StopTime', num2str(simConf.T));
set_param(h.mdl, 'OutputOption', 'SpecifiedOutputTimes');
set_param(h.mdl, 'OutputTimes', sprintf('0:%.17g:%.17g', simConf.Ts, simConf.T));
assignin('base', 't__', h.tspan(:));
assignin('base', 'u__', zeros(numel(h.tspan), numel(simConf.in_name)));
set_param(h.mdl, 'FastRestart', 'on');
end


function discardCopiesOfDeadWorkers(folder)
% The pid in a copy's name changes every run and nothing else removes the old
% ones, so they pile up. Only drop the ones whose process is gone: a copy
% belonging to a concurrently running job must be left alone.
if ~isfolder('/proc') || ~isfolder(folder)
    return;   % pid liveness is only checkable this way on Linux
end
for f = dir(fullfile(folder, 'fr_*_w*.slx'))'
    pid = regexp(f.name, '_w(\d+)(_breach)?\.slx$', 'tokens', 'once');
    if ~isempty(pid) && ~isfolder(['/proc' filesep pid{1}])
        delete(fullfile(folder, f.name));
    end
end
end


function p = weightBlockPath(mdlBreach, oriModelName, layerIdx, rowIdx)
% Same block hierarchy replaceWeights targets, under the instrumented copy's name.
if contains(oriModelName, 'WT') || contains(oriModelName, 'ACC')
    prefix = [mdlBreach '/Feed-Forward Neural Network/Layer ' num2str(layerIdx)];
elseif contains(oriModelName, 'AFC')
    prefix = [mdlBreach '/Air Fuel Control Model 1/Feed-Forward Neural Network/Layer ' num2str(layerIdx)];
elseif contains(oriModelName, 'SC')
    prefix = [mdlBreach '/Subsystem/Feed-Forward Neural Network/Layer ' num2str(layerIdx)];
else
    error('fastRobOpen:unknownModel', 'No known block layout for %s', oriModelName);
end

if layerIdx == 1
    p = [prefix '/IW{1,1}/IW{1,1}(' num2str(rowIdx) ',:)'''];
else
    p = [prefix '/LW{' num2str(layerIdx) ',' num2str(layerIdx-1) '}/IW{' ...
         num2str(layerIdx) ',' num2str(layerIdx-1) '}(' num2str(rowIdx) ',:)'''];
end
end
