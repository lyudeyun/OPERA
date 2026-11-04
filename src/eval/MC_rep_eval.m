function res = MC_rep_eval(ctrl, outFile, coreNum, trFile, teFile)
% MC_rep_eval evaluates an MC controller on the training and test sets.
%
% Safety as in rep_eval_{acc,afc,wt}.py (robustness, number of satisfied cases
% *_SR, executions that turned safe or unsafe), and three performance metrics of
% Mountain Car:
%   reach  - time to reach the goal (position >= 0.45), in steps; the horizon
%            (110) if the goal is not reached. Mountain Car's performance
%            measure since its introduction (Moore 1990; Sutton and Barto 2018,
%            Example 10.1: reward -1 per step until the goal).
%   energy - mean of u^2 per step until the goal (over the whole horizon if it
%            is not reached): the control cost of MountainCarContinuous in
%            OpenAI Gym (Brockman et al. 2016), 0.1*u^2 per step until the goal.
%   smooth - mean of |u(t) - u(t-1)| per step until the goal (over the whole
%            horizon if it is not reached; 0 if the goal is reached at the first
%            step): how much the control signal varies.
% energy and smooth are per-step means, so that they do not grow with the time
% to reach the goal, which reach measures.
% The values before repair come from the trajectories recorded in the datasets,
% so only the given controller is simulated.
%
% Inputs:
%   ctrl: a retrained network (.mat holding net), or a controller in YAML (.yml);
%         name or path
%   outFile: optional; when given, the results are saved there
%   coreNum: optional; number of parallel workers (default: the current pool)
%   trFile, teFile: optional; the training and test sets (test suites with the
%         recorded executions of the original controller); by default MC's
%         MC_Phi_1_FFNN_16_16_XV_100_100_Feb_11_2026_{Tr,Te}.mat
% Outputs:
%   res: struct with the fields saved, e.g. te_SR, te_rob_set, te_reach_set,
%        te_mean_energy, te_neg_to_pos (indices), and the same for tr
%
% Example (the original controller and a network retrained with strategy 5):
%   MC_rep_eval('mc_sig_2x16_broken.yml');
%   MC_rep_eval('MC_FFNN_trainlm_16_16_st_5_config_UnNor_Ep_1000_Feb_15_1.mat', ...
%       'MC_FFNN_trainlm_16_16_st_5_UnNor_Ep_1000_Feb_15_1_Rep_Eval_Result.mat', 10);

dsName = 'MC_Phi_1_FFNN_16_16_XV_100_100_Feb_11_2026';
if nargin < 4 || isempty(trFile), trFile = [dsName '_Tr.mat']; end
if nargin < 5 || isempty(teFile), teFile = [dsName '_Te.mat']; end
ctrlPath = which(ctrl);
if isempty(ctrlPath), ctrlPath = ctrl; end
if endsWith(ctrlPath, '.yml')
    useNet = false;
    net = [];
else
    useNet = true;
    net = load(ctrlPath).net;
    if ~isa(net, 'network')
        error('MC_rep_eval:net', '%s does not hold a network object (saved by a newer MATLAB?)', ctrlPath);
    end
end
if nargin >= 3 && ~isempty(coreNum)
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= coreNum
        delete(pool);
        parpool(coreNum);
    end
end

res = struct();
res.ctrl = ctrlPath;
splits = {'tr', trFile; 'te', teFile};
for si = 1:size(splits, 1)
    s = splits{si, 1};
    dsPath = which(splits{si, 2});
    if isempty(dsPath), dsPath = splits{si, 2}; end
    D = load(dsPath).testSuite;
    n = numel(D.testCases);

    % before repair: the recorded executions of the original controller
    bef = struct();
    [bef.reach, bef.energy, bef.smooth] = arrayfun(@(c) mcMetrics(c.trajPos, c.nnOutputs), D.testCases(:));
    robB = [D.testCases.rob]';

    % after repair: simulate the given controller from the same initial states
    rob = zeros(n, 1); reach = zeros(n, 1); energy = zeros(n, 1); smooth = zeros(n, 1);
    inputs = [D.testCases.sysInputs];
    parfor i = 1:n
        if useNet
            [r, ~, ~, ~, u, pos, ~] = simMatlabMC_Rep(inputs(1,i), inputs(2,i), net, false, zeros(1, 111));
        else
            [r, ~, ~, ~, u, pos, ~] = simMatlabMC(inputs(1,i), inputs(2,i), ctrlPath, false, zeros(1, 111));
        end
        rob(i) = r;
        [reach(i), energy(i), smooth(i)] = mcMetrics(pos, u);
    end

    res.([s '_rob_set']) = rob;
    res.([s '_neg_idx_bef']) = find(robB < 0);
    res.([s '_pos_idx_bef']) = find(robB > 0);
    res.([s '_neg_idx_aft']) = find(rob < 0);
    res.([s '_pos_idx_aft']) = find(rob > 0);
    res.([s '_neg_to_pos']) = find(robB < 0 & rob > 0);
    res.([s '_pos_to_neg']) = find(robB > 0 & rob < 0);
    % numbers of satisfied cases, as rep_eval stores *_SR
    res.([s '_SR_bef']) = numel(find(robB > 0));
    res.([s '_SR']) = numel(find(rob > 0));
    aft = struct('reach', reach, 'energy', energy, 'smooth', smooth);
    for m = {'reach', 'energy', 'smooth'}
        res.([s '_' m{1} '_set_bef']) = bef.(m{1});
        res.([s '_' m{1} '_set']) = aft.(m{1});
        res.([s '_mean_' m{1} '_bef']) = mean(bef.(m{1}));
        res.([s '_mean_' m{1}]) = mean(aft.(m{1}));
    end
    fprintf('%s: SR %.2f%% -> %.2f%% | reach %.2f -> %.2f | energy %.3f -> %.3f | smooth %.3f -> %.3f\n', s, ...
        100 * res.([s '_SR_bef']) / n, 100 * res.([s '_SR']) / n, res.([s '_mean_reach_bef']), res.([s '_mean_reach']), ...
        res.([s '_mean_energy_bef']), res.([s '_mean_energy']), res.([s '_mean_smooth_bef']), res.([s '_mean_smooth']));
end
if nargin >= 2 && ~isempty(outFile)
    save(outFile, '-struct', 'res');
end
end

function [reach, energy, smooth] = mcMetrics(trajPos, u)
% trajPos(1) is the state at time 0, so the goal is reached at step k when
% trajPos(k+1) >= 0.45; the controls applied until then are u(1:k).
horizon = numel(u);
k = find(trajPos(2:end) >= 0.45, 1);
if isempty(k)
    reach = horizon;
    k = horizon;
else
    reach = k;
end
uk = u(1:k);
energy = mean(uk .^ 2);
if k > 1
    smooth = mean(abs(diff(uk)));
else
    smooth = 0;
end
end
