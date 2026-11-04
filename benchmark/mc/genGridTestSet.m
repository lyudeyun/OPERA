function testSuite = genGridTestSet(gridNumPos, gridNumVel, seed, outFile, nnFile, posRange, velRange)
% genGridTestSet simulates MC on a seeded grid of initial states.
%
% One random initial state in every grid cell (gridSampling2D), with rng(seed)
% set first so that the result can be reproduced. test/gen_dataset.py calls it
% for config/MC/gendata1, which generates both the retraining set (Tr) and the
% test set (Te) of MC. (The Tr in the repository was generated before, on the
% same 100 x 100 grid but without a seed.)
%
% Inputs:
%   gridNumPos, gridNumVel: grid cells along position and velocity
%   seed: seed of the random states inside the cells
%   outFile: optional; when given, testSuite is saved there
%   nnFile: optional; the controller (default mc_sig_2x16_broken.yml)
%   posRange, velRange: optional; the ranges of the initial position and velocity
%                       (default [-0.505, 0.395] and [-0.055, 0.045])
% Outputs:
%   testSuite: struct with sn, size and testCases (sysInputs, nnInputs, nnOutputs,
%              rob, T_l, T_r, trajPos, trajVel)
%
% Example (the Te of MC, 1000 traces):
%   genGridTestSet(40, 25, 0, 'benchmark/mc/MC_Phi_1_FFNN_16_16_XV_100_100_Feb_11_2026/dataset/MC_Phi_1_FFNN_16_16_XV_100_100_Feb_11_2026_Te.mat');

if nargin < 5 || isempty(nnFile), nnFile = 'mc_sig_2x16_broken.yml'; end
if nargin < 6 || isempty(posRange), posRange = [-0.505, 0.395]; end
if nargin < 7 || isempty(velRange), velRange = [-0.055, 0.045]; end
testCaseNum = gridNumPos * gridNumVel;

rng(seed);
samples = gridSampling2D(posRange(1), posRange(2), velRange(1), velRange(2), gridNumPos, gridNumVel);

% found on the path, so this runs from any folder
nnPath = which(nnFile);
if isempty(nnPath), nnPath = nnFile; end
testCases(testCaseNum, 1) = struct('sysInputs', [], 'nnInputs', [], 'nnOutputs', [], 'rob', [], 'T_l', [], 'T_r', [], 'trajPos', [], 'trajVel', []);
parfor tci = 1:testCaseNum
    [rob, T_l, T_r, traj_nn_in, traj_nn_out, traj_pos, traj_vel] = simMatlabMC(samples(tci,1), samples(tci,2), nnPath, false, zeros(1,111));
    testCases(tci).sysInputs = samples(tci,:)';
    testCases(tci).nnInputs = traj_nn_in;
    testCases(tci).nnOutputs = traj_nn_out;
    testCases(tci).rob = rob;
    testCases(tci).T_l = T_l;
    testCases(tci).T_r = T_r;
    testCases(tci).trajPos = traj_pos;
    testCases(tci).trajVel = traj_vel;
end

testSuite.sn = sum([testCases.rob] > 0);
testSuite.size = testCaseNum;
testSuite.testCases = testCases;
testSuite.seed = seed;
testSuite.sampling = sprintf('grid %d x %d over pos %s, vel %s, one random state per cell, rng(%d)', ...
    gridNumPos, gridNumVel, mat2str(posRange), mat2str(velRange), seed);
if nargin >= 4
    save(outFile, 'testSuite');
end
fprintf('%d satisfying, %d violating of %d\n', testSuite.sn, testCaseNum - testSuite.sn, testCaseNum);
end
