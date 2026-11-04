function [rob, T_l, T_r, traj_nn_in, traj_nn_out, trajPos, trajVel] = simMatlabMC_Rep(initPos, initVel, net, patchFlag, patch)
% SIMMATLABMC_REP Simulates the Mountain Car (MC) system using a MATLAB neural network object.
%
% Inputs:
%   initPos  - Initial position of the car
%   initVel  - Initial velocity of the car
%   net      - MATLAB network object
%   patchFlag- Apply patch or not
%   patch    - The patch applied to the original control output, all zero or 0000xxxxx000
%
% Outputs:
%   rob             - STL robustness value: max(trajPosX) - 0.45
%   score           - 
%   T_l             - the time when the violation episode appear
%   T_r             - the time when the violation episode disappear
%   traj_nn_in      - Position and Velocity inputs to NN at each step (2×110)
%   traj_nn_out     - NN outputs (control signals) at each step (1×110)
%   trajPos         - Position trajectory of the car (1×111)
%   trajVel         - Velocity trajectory of the car (1×111)

steepness = 0.0025;    % Slope of the hill
len = 111;             % Simulation length (0 to 110)

% Initialize trajectories
trajPos = zeros(1, len);
trajVel = zeros(1, len);
traj_nn_in_1 = zeros(1, len - 1);
traj_nn_in_2 = zeros(1, len - 1);
traj_nn_out = zeros(1, len - 1);

% Set initial state
trajPos(1) = initPos;
trajVel(1) = initVel;

for i = 2:len
    % Update position using previous velocity
    trajPos(i) = trajPos(i-1) + trajVel(i-1);

    % Prepare NN inputs: [position; velocity]
    nn_inputs = [trajPos(i-1); trajVel(i-1)];

    % Record inputs for logging
    traj_nn_in_1(i-1) = trajPos(i-1);
    traj_nn_in_2(i-1) = trajVel(i-1);

    % NN forward pass using MATLAB network object
    nn_out = net(nn_inputs);

    if patchFlag
        % Since the patch has the same length as nn_out, 
        % the patch value corresponding to the current time step 
        % can be applied accordingly.
        nn_out = nn_out + patch(i);
    end

    % Record NN output
    traj_nn_out(i-1) = nn_out;

    % Update velocity based on physics and control signal
    trajVel(i) = trajVel(i-1) + 0.0015 * nn_out - steepness * cos(3 * trajPos(i-1));

    % Velocity saturation
    if trajVel(i) > 0.07
        trajVel(i) = 0.07;
    elseif trajVel(i) < -0.07
        trajVel(i) = -0.07;
    end

    % Position saturation
    if trajPos(i) > 0.6
        trajPos(i) = 0.6;
    elseif trajPos(i) < -1.2
        trajPos(i) = -1.2;
    end

    % Edge condition: reset velocity if hitting the leftmost boundary
    if trajPos(i) == -1.2 && trajVel(i) < 0
        trajVel(i) = 0;
    end
end

traj_nn_in = [traj_nn_in_1; traj_nn_in_2];

% STL robustness: ensure position eventually reaches ≥ 0.45
rob = max(trajPos) - 0.45;

if rob > 0
    T_l = NaN; 
    T_r = NaN;
else
    % This is the property of eventually
    T_l = 110;
    T_r = 110;
end
end
