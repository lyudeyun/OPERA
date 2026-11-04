function [rob, taus, traj_nn_in, traj_nn_out, traj_pos, traj_vel] = simPythonMC(initPos, initVel, nnPath)
% SIMPYTHONMC Simulates the Mountain Car system by invoking a Python script.
%
% Inputs:
%   initPos - Initial position of the car (must be in [-0.505, 0.395])
%   initVel - Initial velocity of the car (must be in [-0.055, 0.045])
%   nnPath  - Path to the YAML file with NN weights and biases
%
% Outputs:
%   rob         - STL robustness: max(traj_pos) - 0.45
%   taus        - Time horizon of the STL specification (should be 110)
%   traj_nn_in  - [2 × T] NN inputs at each step: [pos; vel]
%   traj_nn_out - [1 × T] NN output (control signal)
%   traj_pos    - [1 × (T+1)] Position trajectory
%   traj_vel    - [1 × (T+1)] Velocity trajectory

    % Input validation
    if nargin < 3
        error('initPos, initVel, and nnPath must be provided!');
    end

    % Check input ranges (domain of valid simulation)
    isPosValid = initPos >= -0.505 && initPos <= 0.395;
    isVelValid = initVel >= -0.055 && initVel <= 0.045;

    if ~isPosValid || ~isVelValid
        fprintf('Invalid input: initPos = %.2f, initVel = %.2f\n', initPos, initVel);
        error('Input parameters are out of the valid range!');
    end

    % Paths relative to this M-file so script works regardless of MATLAB cwd
    mcDir = fileparts(mfilename('fullpath'));
    projRoot = fileparts(fileparts(mcDir));  % OPERA_dev
    pyScript = fullfile(mcDir, 'sim_mc.py');
    % Run Python from project root so nnPath like 'benchmark/mc/...yml' resolves
    cmd = sprintf('cd "%s" && python3 "%s" --init_pos %.6f --init_vel %.6f --nn_path "%s"', ...
                  projRoot, pyScript, initPos, initVel, nnPath);

    % Execute the command
    [status, cmdout] = system(cmd);

    if status == 0
        % Parse JSON output from Python
        diagInfo = jsondecode(cmdout);

        % Extract and transpose trajectory data
        rob          = diagInfo.rob;
        taus         = diagInfo.taus;
        traj_nn_in_1 = diagInfo.traj_nn_in_1';  % 1 × T
        traj_nn_in_2 = diagInfo.traj_nn_in_2';  % 1 × T
        traj_nn_out  = diagInfo.traj_nn_out';   % 1 × T
        traj_pos     = diagInfo.traj_pos';      % 1 × (T+1)
        traj_vel     = diagInfo.traj_vel';      % 1 × (T+1)

        % Combine NN input into a 2×T array
        traj_nn_in   = [traj_nn_in_1; traj_nn_in_2];

    else
        % Print Python stderr if execution fails
        error('MC simulation failed with error:\n%s', cmdout);
    end
end