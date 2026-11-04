function [rob, tout, X] = fastRob(h, sysInputs)
% FASTROB  STL robustness of one test case: native sim() for the trajectory,
% Breach's STL engine for the robustness.
%
% tout and X optionally return the trajectory the robustness was computed on
% (columns of X follow h.sigNames). Both are empty when the simulation failed,
% so a caller that needs the trajectory can tell a failure apart from a real
% robustness of -1e6 instead of reading a stale trace.

% Every sim leaves a run in the Simulink Data Inspector, about 3 MB each, and
% turning signal-log inspection off does not stop it -- which is why Breach
% clears SDI on every call. Clearing costs ~46 ms, more than a simulation, so
% do it in batches instead: well under 1 ms per sim, at the price of roughly
% 150 MB resident per worker (measured over 900 sims), so budget for that
% times the pool size rather than once per job.
persistent sinceClear
if isempty(sinceClear), sinceClear = 0; end
sinceClear = sinceClear + 1;
if sinceClear >= 50
    Simulink.sdi.clear;
    sinceClear = 0;
end

if numel(sysInputs) ~= numel(h.in_name)
    error('fastRob:inputCount', 'got %d input signals, the model takes %d', ...
        numel(sysInputs), numel(h.in_name));
end

u = zeros(numel(h.tspan), numel(h.in_name));
for i = 1:numel(h.in_name)
    cpi = h.cp(i);
    if cpi == 1
        % a single control point is a constant signal; interp1 needs two
        u(:, h.portOfInput(i)) = sysInputs{i}(1);
    else
        t_cp = linspace(h.tspan(1), h.tspan(end), cpi + 1)';
        u(:, h.portOfInput(i)) = interp1(t_cp(1:end-1), sysInputs{i}(:), h.tspan(:), 'previous', 'extrap');
    end
end
assignin('base', 'u__', u);

% The search suspends the session whenever it finishes, so fast restart may be
% off; turning it back on is far cheaper than rebuilding the session.
if strcmp(get_param(h.mdl, 'FastRestart'), 'off')
    set_param(h.mdl, 'FastRestart', 'on');
end

% Simulation may fail because an unstable controller diverges the solver, which
% is a normal event here: such a candidate simply scores as unrepaired. A
% failure does not leave the session unusable -- the next simulation returns
% exactly what a freshly opened session would.
%
% A signal or input that does not line up with the specification is a different
% matter: that is a misconfigured experiment, and swallowing it would make every
% candidate look like it diverged. Let those out.
tout = [];
X = [];
try
    so = sim(h.mdl);
    [tout, X] = extractSignals(so, h.sigNames, h.outportNames);
    h.trace.ResetSimulations();
    h.trace.AddTrace([tout, X]);
    rob = h.trace.CheckSpec(h.phi);
catch ME
    if startsWith(ME.identifier, 'fastRob:')
        rethrow(ME);
    end
    rob = -1e6;
    tout = [];
    X = [];
end
end


function [tout, X] = extractSignals(so, sigNames, outportNames)
% ACC models log to `logsout`; AFC models have signal logging off and route
% signals through Outports into `yout`, where the labels come back empty.
% Breach resolves a name present in both in favour of logsout, so do the same
% rather than letting so.who's ordering decide.
tout = so.tout;
X = nan(numel(tout), numel(sigNames));
filled = false(1, numel(sigNames));

vars = cellstr(so.who);
for pass = 1:2
    for vi = 1:numel(vars)
        if strcmp(vars{vi}, 'tout'), continue; end
        v = so.get(vars{vi});
        isDataset = isa(v, 'Simulink.SimulationData.Dataset');
        if (pass == 1) == isDataset, continue; end   % structs first, datasets second

        if isDataset
            present = v.getElementNames();
            for k = 1:numel(sigNames)
                if ~any(strcmp(present, sigNames{k})), continue; end
                el = v.get(sigNames{k});
                X(:,k) = interp1(el.Values.Time, el.Values.Data, tout, 'linear', 'extrap');
                filled(k) = true;
            end
        elseif isstruct(v) && isfield(v, 'signals') && isfield(v, 'time')
            for si = 1:numel(v.signals)
                lbl = v.signals(si).label;
                if isempty(lbl) && si <= numel(outportNames)
                    lbl = outportNames{si};
                end
                k = find(strcmp(sigNames, lbl), 1);
                if ~isempty(k) && ~filled(k)
                    X(:,k) = interp1(v.time, v.signals(si).values, tout, 'linear', 'extrap');
                    filled(k) = true;
                end
            end
        end
    end
end

if ~all(filled)
    error('fastRob:signalNotFound', 'signals not produced by the model: %s', ...
        strjoin(sigNames(~filled), ', '));
end
end
