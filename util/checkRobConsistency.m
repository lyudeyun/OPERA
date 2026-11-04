function [is_consistent, is_opposite_sign] = checkRobConsistency(init_rob, orig_rob)
% checkRobConsistency checks whether the recalculated robustness matches
% the original robustness stored in the dataset.
%
% Inputs:
%   init_rob: robustness recalculated by the current simulation
%   orig_rob: original robustness from the dataset
% Outputs:
%   is_consistent: true if consistent, false otherwise
%   is_opposite_sign: true if init_rob and orig_rob have opposite signs
%
% Rules (hardcoded thresholds):
%   1) opposite signs -> always mismatch (callers should error)
%   2) same sign, at least one |rob| < 1 -> absolute threshold 0.01
%   3) same sign, both |rob| >= 1 -> relative threshold 0.05
%
% On mismatch, prints both robustness values before returning false.

is_opposite_sign = (init_rob > 0 && orig_rob < 0) || (init_rob < 0 && orig_rob > 0);
is_consistent = true;

if is_opposite_sign
    is_consistent = false;
elseif abs(init_rob) < 1 || abs(orig_rob) < 1
    if abs(init_rob - orig_rob) >= 0.01
        is_consistent = false;
    end
else
    if abs(init_rob - orig_rob) / abs(orig_rob) >= 0.05
        is_consistent = false;
    end
end

if ~is_consistent
    fprintf('init_rob (calculated): %.16g\n', init_rob);
    fprintf('orig_rob (dataset):    %.16g\n', orig_rob);
    fprintf('abs(diff):             %.16g\n', abs(init_rob - orig_rob));
end
end
