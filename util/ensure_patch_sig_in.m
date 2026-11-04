function ensure_patch_sig_in()
% Rebuild the From Workspace table used by RepCont-style models:
%   plant <- NN + patch_sig + safe_patch_sig
% Keep the block's Interpolate option on so an all-zero patch does not add
% discrete sample hits to the continuous plant.  For a nonzero patch, encode
% exact zero-order-hold jumps with duplicate timestamps:
%   [t_k, old_value; t_k, new_value].
% This preserves the old discrete patch waveform without reintroducing the
% zero-patch solver perturbation caused by a discrete source/Switch model.
% All-zero patches collapse to two points.
% Prefer the Opera globals; fall back to base workspace copies.
% Call this before BreachSimulinkSystem / Sim so the table is not stale.
if ~evalin('base', 'exist(''T'',''var'')') || ~evalin('base', 'exist(''Ts'',''var'')')
	return;
end
T_ = evalin('base', 'T');
Ts_ = evalin('base', 'Ts');
t_ = (0:Ts_:T_)';

global patch_sig safe_patch_sig;
ps_ = zeros(size(t_));
ss_ = zeros(size(t_));
ps_src = [];
ss_src = [];
if ~isempty(patch_sig)
	ps_src = patch_sig(:);
elseif evalin('base', 'exist(''patch_sig'',''var'')')
	ps_src = evalin('base', 'patch_sig');
	ps_src = ps_src(:);
end
if ~isempty(safe_patch_sig)
	ss_src = safe_patch_sig(:);
elseif evalin('base', 'exist(''safe_patch_sig'',''var'')')
	ss_src = evalin('base', 'safe_patch_sig');
	ss_src = ss_src(:);
end
if numel(ps_src) == numel(t_)
	ps_ = ps_src;
end
if numel(ss_src) == numel(t_)
	ss_ = ss_src;
end
y_ = ps_ + ss_;
if all(y_ == 0)
	assignin('base', 'patch_sig_in', [0 0; T_ 0]);
else
	% Piecewise-constant table for a From Workspace block with
	% Interpolate='on'. Duplicate times describe an instantaneous jump.
	change_idx = find(diff(y_) ~= 0) + 1;
	stair_t = zeros(1 + 2*numel(change_idx) + 1, 1);
	stair_y = zeros(size(stair_t));
	row = 1;
	stair_t(row) = t_(1);
	stair_y(row) = y_(1);
	for k = change_idx(:)'
		row = row + 1;
		stair_t(row) = t_(k);
		stair_y(row) = y_(k-1);
		row = row + 1;
		stair_t(row) = t_(k);
		stair_y(row) = y_(k);
	end
	if stair_t(row) < T_
		row = row + 1;
		stair_t(row) = T_;
		stair_y(row) = y_(end);
	end
	assignin('base', 'patch_sig_in', [stair_t(1:row) stair_y(1:row)]);
end
end
