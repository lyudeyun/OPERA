function verify_parpool(core_num)
% Quick check that parpool can start with the requested worker count.
% Usage (from repo root):
%   matlab -batch "addpath('util'); verify_parpool(20);"

if nargin < 1
    core_num = 20;
end

fprintf('=== parpool verification ===\n');
fprintf('Requested workers: %d\n', core_num);
fprintf('feature(''numcores''): %d\n', feature('numcores'));

cluster = parcluster('local');
fprintf('local cluster NumWorkers (max): %d\n', cluster.NumWorkers);

existing = gcp('nocreate');
if ~isempty(existing)
    delete(existing);
end

pool = parpool('local', core_num);
fprintf('Started pool workers: %d\n', pool.NumWorkers);

t_start = tic;
parfor i = 1:core_num
    pause(2);
end
elapsed = toc(t_start);
fprintf('parfor pause(2) x %d elapsed: %.2f s\n', core_num, elapsed);
fprintf('(expect ~2 s if workers run in parallel; ~%d s if serial)\n', core_num * 2);

delete(pool);
fprintf('=== verification done ===\n');

end
