classdef pyOpera < handle
    properties
        bm
        nn_bef
        nn_rep
        D
        T
        Ts
        T_l
        T_r
        in_name
        in_range
        in_span % 如果input不是信号，是常数，那么in_span的长度就跟T一样长。
        icc_name
        ics_name
        ic_const
        ic_sig
        ocb_name
        ocm_name
        ocf_name
        ocb_sig
        ocf_sig
        oc_range
        oc_span
        sw_sz
        phi 
        pop_size
        max_gen
        M
        V
        N
        use_custom_patch_points
        amp
        patch_range
        st_id
        opt_pos         % the customized start time for optimization (narrow down the search space)

        t_l_ratio
        t_r_ratio
        use_custom_search_range
        chroflag

        chro_prefix
        rep_budget

        % %% model
        % bm              % benchmark
        % mdl_bef
        % mdl_rep
        % D               % the dataset to be repaired
        % T               % simulation time
        % Ts              % sampling interval
        % T_l             % the first instance when the robustness value turns negative
        % T_r             % the first instance when the negative robustness dispears or the instance when the simulation ends
        % %% the input signal of the system
        % in_name         % the input signals of the system
        % in_range        % the range of the input signals
        % in_span         % the time span of the input signals
        % %% the input signal of the controller
        % icc_name        % the input constant of the controller
        % ics_name        % the input signals of the controller
        % ic_const        % the input constant value of the controller
        % ic_sig          % the input signal values of the controller
        % %% the output signal of the controller
        % ocb_name        % the output signal of the controller before repair
        % ocm_name        % the mixed output signal controlled by a plus block (patch_sig + a_ego_b)
        % ocf_name        % the final output signal controlled by sw
        % ocb_sig         % the value of the output signal of the controller before repair
        % ocf_sig         % the value of the final output signal controlled by sw
        % oc_range        % the range of the output signal
        % oc_span         % the time span of the output signal
        % %% specification parameters
        % phi             % the specification in the form of STL
        % phic            % the customized specification
        % phic_str        % the string of the customized specification
        % interval_LR     % the interval specified in the specification (for intersection)
        % %% multi-objective optimization parameters
        % pop_size        % population size
        % max_gen         % maximum number of the generation
        % M               % the number of the objectives
        % V               % the number of the decision variables
        % N               % the number of the interpolation points
        % use_custom_patch_points  % whether to use custom patch points setting (vary N/V) when use_custom_search_range is false
        % amp             % the ampultute of the control signal patch
        % patch_range     % the range of the control signal patch
        % st_id           % strategy id
        % opt_pos         % the customized start time for optimization (narrow down the search space)
        % t_l_ratio       % ratio for t_l lower bound in [opt_pos, T_l] range (0-1), where t_l_lb = opt_pos + t_l_ratio * (T_l - opt_pos)
        % t_r_ratio       % ratio for t_r upper bound in [T_l, T_r] range (0-1), where t_r_ub = T_r - t_r_ratio * (T_r - T_l)
        % use_custom_search_range  % whether to use custom search range (t_l_ratio and t_r_ratio) or use default (opt_pos to T_l/T_r)
        % chroflag        % exist chromosome files or not
        % chro_prefix     % chromosome prefix
        % rep_budget      % repair budget
    end
    methods
        function this = pyOpera(bm, nn_bef, nn_rep, D, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, ocb_name, ocm_name, ocf_name, oc_range, oc_span, phi, pop_size, max_gen, M, V, N, use_custom_patch_points, amp, st_id, opt_pos, use_custom_search_range, t_l_ratio, t_r_ratio, chro_prefix, rep_budget)
            this.bm = bm;
            this.nn_bef = nn_bef;
            this.nn_rep = nn_rep;
            this.D = D;

            this.T = T;
            this.Ts = Ts;
            this.T_l = 0;
            this.T_r = 0;

            this.in_name = in_name;
            this.in_range = in_range;
            this.in_span = cell2mat(in_span);

            this.icc_name = icc_name;
            this.ic_const = ic_const;
            this.ics_name = ics_name;
            this.ic_sig = [];

            this.ocb_name = ocb_name;
            this.ocm_name = ocm_name;
            this.ocf_name = ocf_name;
            this.ocb_sig = [];
            this.ocf_sig = [];
            this.oc_range = oc_range;
            this.oc_span = cell2mat(oc_span);
            this.sw_sz = T/Ts+1;
            this.phi = phi;

            this.pop_size = pop_size;
            this.max_gen = max_gen;
            this.M = M;
            this.V = V;
            this.N = N;
            if this.N < 3
                error('N must be >= 3 (endpoints fixed to 0, so at least one free interpolation point is required).');
            end
            this.use_custom_patch_points = use_custom_patch_points;
            this.amp = amp;
            this.patch_range = {};
            for i = 1:numel(this.oc_range)
                if this.oc_range{1,i}(1,1) < 0 && this.oc_range{1,i}(1,2) > 0
                    this.patch_range{1,end+1} = this.amp * this.oc_range{1,i};
                elseif all(this.oc_range{1,i} >= 0) || all(this.oc_range{1,i} <= 0)
                    delta_range = this.oc_range{1,i}(1,2) - this.oc_range{1,i}(1,1);
                    this.patch_range{1,end+1} = [-0.5 * this.amp * delta_range, 0.5 * this.amp * delta_range];
                else
                    error('Check the oc_range!');
                end
            end
            this.st_id = st_id;
            this.opt_pos = opt_pos;

            this.use_custom_search_range = use_custom_search_range;
            this.t_l_ratio = t_l_ratio;
            this.t_r_ratio = t_r_ratio;
            if this.use_custom_search_range && this.use_custom_patch_points
                error('Invalid configuration: use_custom_search_range and use_custom_patch_points cannot both be true.');
            end
            if this.use_custom_search_range && t_l_ratio == 1 && t_r_ratio == 1
                error('Invalid search range configuration: t_l_ratio = 1 and t_r_ratio = 1.');
            end
            % When custom search range is enabled, enforce a fixed patch encoding (V=6, N=4)
            if this.use_custom_search_range
                this.V = 6;
                this.N = 4;
            end
            this.chro_prefix = chro_prefix;
            this.rep_budget = rep_budget;
        end
        %% functions
        function [rob, scores, T_l, T_r, traj_nn_in, traj_nn_out, traj_pos_x, traj_pos_y] = simMatlabUUV(this, init_pos_y, init_global_heading_deg, nnPath, patchFlag, patch_sig)
            % MATLABUUV Simulates UUV dynamics using a trained neural network controller.
            %
            % Inputs:
            %   init_pos_y              - Initial Y-axis position of the UUV
            %   init_global_heading_deg - Initial global heading (degrees)
            %   nnPath                  - Path to the YAML file storing the trained NN
            %   patchFlag               - Apply patch or not
            %   patch                   - The patch applied to the original control output, all zero or 0000xxxxx000
            %
            % Outputs:
            %   rob           - STL robustness value (pos_y should stay within [10, 50])
            %   score         -
            %   T_l           - the time when the violation episode appear
            %   T_r           - the time when the violation episode disappear
            %   traj_nn_in    - NN input: pipe heading and starboard range at each step
            %   traj_nn_out   - NN output: heading adjustment command at each step
            %   traj_pos_x    - UUV X position trajectory
            %   traj_pos_y    - UUV Y position trajectory

            % Load neural network from YAML
            uuvNet = convertUUVYamlToMat(nnPath);

            % Load UUV dynamic model (discretized)
            mat_path = which('uuv_model_oneHz.mat');
            data = load(mat_path);
            A = data.A; B = data.B; C = data.C; D = data.D;

            % Initial heading in radians
            init_global_heading = deg2rad(init_global_heading_deg);

            % Initial system state: [heading; velocity; ...]
            x = [0; 0; 0; 0];

            % Initial control input: [heading; surge; dive]
            u = [0; 0.48556; 45.0];

            % Initial position
            pos_x = 0.0;
            pos_y = init_pos_y;

            % Initialize trajectory logs
            traj_pos_x = pos_x;
            traj_pos_y = pos_y;
            traj_nn_in_1 = [];
            traj_nn_in_2 = [];
            traj_nn_out = [];

            for i = 1:30
                % System dynamics: y = output, x = updated state
                y = C * x + D * u;
                x = A * x + B * u;

                % Extract heading and calculate global heading
                heading = y(1);
                if heading > pi
                    heading = heading - 2 * pi;
                end
                global_heading = heading + init_global_heading;

                % Update position
                pos_x = pos_x + y(2) * cos(global_heading);
                pos_y = pos_y - y(2) * sin(global_heading);
                traj_pos_x(end + 1) = pos_x;
                traj_pos_y(end + 1) = pos_y;

                % NN inputs
                pipe_heading = -global_heading;
                stdb_range = pos_y / cos(global_heading);
                nn_inputs = [pipe_heading; stdb_range];

                % NN inference (3-layer tanh NN)
                z1 = uuvNet.IW{1,1} * nn_inputs + uuvNet.b{1};
                a1 = tanh(z1);
                z2 = uuvNet.LW{2,1} * a1 + uuvNet.b{2};
                a2 = tanh(z2);
                z3 = uuvNet.LW{3,2} * a2 + uuvNet.b{3};
                nn_out = tanh(z3);

                if patchFlag
                    % Since the patch has the same length as nn_out,
                    % the patch value corresponding to the current time step
                    % can be applied accordingly.
                    nn_out = nn_out + patch_sig(i+1);
                end

                traj_nn_in_1(end + 1) = pipe_heading;
                traj_nn_in_2(end + 1) = stdb_range;
                traj_nn_out(end + 1) = nn_out;

                % % Early termination if out of bounds
                % if pos_y < 10 || pos_y > 50 || pos_x < -10 || pos_x > 400
                %     break;
                % end

                % Update heading input based on NN output
                heading_delta = deg2rad(5) * nn_out;
                abs_heading = heading + heading_delta;
                if abs_heading > pi
                    abs_heading = abs_heading - 2 * pi;
                end
                u = [abs_heading; 0.48556; 45.0]; % surge and dive fixed
            end

            traj_nn_in = [traj_nn_in_1; traj_nn_in_2];

            % STL robustness: G_[0,30](pos_y ∈ [10, 50])
            scores = arrayfun(@(y) min(50 - y, y - 10), traj_pos_y);
            rob = min(scores);

            if rob > 0
                T_l = NaN;
                T_r = NaN;
            else
                T_l = find(scores < 0, 1, 'first') - 1;
                isT_r = find(diff(scores < 0) == -1, 1);
                if isT_r
                    T_r = isT_r - 1;
                else
                    T_r = 30;
                end
            end
        end


        function [rob, T_l, T_r, traj_nn_in, traj_nn_out, trajPos, trajVel] = simMatlabMC(this, initPos, initVel, nnPath, patchFlag, patch_sig)
            % SIMMATLABMC Simulates the Mountain Car (MC) system using a neural network controller.
            %
            % Inputs:
            %   initPos  - Initial position of the car
            %   initVel  - Initial velocity of the car
            %   nnPath   - Path to the YAML file containing the NN weights and biases
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

            % Load neural network parameters from YAML
            mcNet = convertMCYamlToMat(nnPath);

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

                % NN forward pass
                z1 = mcNet.IW{1,1} * nn_inputs + mcNet.b{1};
                a1 = this.sigmoid(z1);
                z2 = mcNet.LW{2,1} * a1 + mcNet.b{2};
                a2 = this.sigmoid(z2);
                z3 = mcNet.LW{3,2} * a2 + mcNet.b{3};
                nn_out = tanh(z3);  % Output control signal

                if patchFlag
                    % Since the patch has the same length as nn_out,
                    % the patch value corresponding to the current time step
                    % can be applied accordingly.
                    nn_out = nn_out + patch_sig(i);
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

        % Sigmoid activation function
        function y = sigmoid(this, x)
            y = 1 ./ (1 + exp(-x));
        end

        function [y, cons] = repairObjFun(this, x, input_1, input_2)
            % Given an external input signal of the system and an individual, repairObjFun function returns the corresponding y and cons value.
            % repairObjFun function defines three objective functions:
            %   1) satisfy the given specification;
            %   2) minimize the time interval [t_l, t_r];
            %   3) minimize the difference bewteen the original control signal and the synthesized control signal, i.e., minimize the interference to the control decision.
            %
            % Inputs:
            %   x: an array of the decision variables, x = [t_l, t_r, the interpolation points to generate the control signal patch]
            %   in_sig: an external input signal of the system
            % Outputs:
            %   y: an array including the values of three objective functions for current individual
            %   cons: a constraint violation logger

            % initialize y and cons
            y = [];
            cons = [];

            % calculate the constraint violations
            c = x(2) - x(1);
            if c < 0
                cons(1) = abs(c);
            end

            % round to the nearest tenth
            t_l = roundn(x(1),0);
            t_r = roundn(x(2),0);

            % a hard constraint: t_l < t_r
            if t_l >= t_r || c < 0
                y = [inf,inf,inf];
                disp(y);
                % reset patch_sig
                patch_sig = zeros(1, this.sw_sz);
                return;
            end

            % the interpolation points to generate the control signal patch (N>=3: endpoints fixed to 0, so at least 1 free point)
            disc_patch = x(3:this.V);
            spacing = (t_r-t_l)/(this.N-1);
            t_control = t_l:spacing:t_r;
            v_control = disc_patch;
            if (t_r - t_l)/this.Ts >= 3
                inp_patch_sig = interp1(t_control, v_control, t_l:this.Ts:t_r, 'makima');
            else
                inp_patch_sig = mean(v_control) * ones(1, int32((t_r - t_l)/this.Ts) + 1);
            end

            % constrain the range of the control signal (here we only consider the case where the controller output dimension is 1)
            % obtain the idx exceeding the upper bound
            ub_i = inp_patch_sig(1,:) > this.patch_range{1,1}(1,2);
            inp_patch_sig(1,ub_i) = this.patch_range{1,1}(1,2);
            % obtain the idx exceeding the lower bound
            lb_i = inp_patch_sig(1,:) < this.patch_range{1,1}(1,1);
            inp_patch_sig(1,lb_i) = this.patch_range{1,1}(1,1);

            % reset sw, original nn controller: [0, this.T]; patch_sig: [t_l, t_r]
            pos_l = int32(t_l/this.Ts)+1;
            pos_r = int32(t_r/this.Ts)+1;

            patch_sig = zeros(1, this.sw_sz);
            patch_sig(1, pos_l:pos_r) = inp_patch_sig;

            % objective function 1
            if strcmp(this.bm, 'uuv')
                [rob, ~, ~, ~, ~, ~, ~, ~] = this.simMatlabUUV(input_1, input_2, 'benchmark/uuv/uuv_tanh_2_15_2x32_broken.yml', true, patch_sig);
            elseif strcmp(this.bm, 'mc')
                [rob, ~, ~, ~, ~, ~, ~] = this.simMatlabMC(input_1, input_2, 'benchmark/mc/mc_sig_2x16_broken.yml', true, patch_sig);
            else
                error('Check your benchmark!');
            end

            % objective function 2
            dist = t_r - t_l;
            % objective function 3
            sig_diff = sum(abs(inp_patch_sig));

            % output
            y = [-rob, dist, sig_diff];
            disp(y);

            % reset patch_sig
            patch_sig = zeros(1, this.sw_sz);
        end

        function [traj_nn_in, traj_nn_out, new_rob, rep_log] = sigRepair(this, input_1, input_2, chroflag, in_i)
            % sigRepair function can synthesize the control signal for a given external input signal of the system.
            %
            % Inputs:
            %   input_1: a given input signal of the system
            %   input_2: a given input signal of the system
            %   chroflag: exist chromosome or not (applicable only to chromosomes of the first round of repair). If it exists, reuse it; otherwise, generate it
            %   in_i: the idx of in_sig, used for visualizing the repair progress
            % Outputs:
            %   traj_nn_in: the repaired input signal of the nn controller
            %   traj_nn_out: the repaired output signal of the nn controller
            %   new_rob: new robustness value after repair
            %   rep_log: repair log, e.g. {[this.T_l, this.T_r, repair_count]}

            safe_patch_sig = zeros(1, this.sw_sz);
            % initialize the repair log
            rep_log = {};
            
            
            if strcmp(this.bm, 'uuv')
                % recheck the STL robustness: G_[0,30](pos_y ∈ [10, 50])
                % perform signal diagnosis
                % get the control signal of the nn controller to be repaired
                [init_rob, ~, this.T_l, this.T_r, ~, this.ocb_sig, ~, ~] = this.simMatlabUUV(input_1, input_2, 'benchmark/uuv/uuv_tanh_2_15_2x32_broken.yml', false, zeros(1,this.sw_sz));
            elseif strcmp(this.bm, 'mc')
                [init_rob, this.T_l, this.T_r, ~, this.ocb_sig, ~, ~] = this.simMatlabMC(input_1, input_2, 'benchmark/mc/mc_sig_2x16_broken.yml', false, zeros(1,this.sw_sz));
            else
                error('Check your benchmark!');
            end

            % simulation error
            if init_rob > 0
                new_rob = init_rob;
                rep_log{1,end+1} = [NaN,NaN,0];
                disp('The robustness value has been corrected!');
                return;
            elseif abs(init_rob - this.D.testSuite.testCases(in_i).rob) >= 0.01
                error('The calculated robustness value does not match the original robustness value.');
            end

            % count the number of the nsga-ii executions
            repair_count = 0;

            % loop until the robustness value of this.phi turns postive or the repair budget is reached
            while true
                % generate the chromosome's name based on the index
                % If using custom search range, include t_l_ratio and t_r_ratio in the filename.
                % If using custom patch points (and NOT using custom search range), include N in the filename.
                if this.use_custom_search_range
                    chro_name = [this.chro_prefix, '_t_l_ratio_', num2str(this.t_l_ratio), '_t_r_ratio_', num2str(this.t_r_ratio), '_', num2str(in_i), '.mat'];
                elseif this.use_custom_patch_points
                    chro_name = [this.chro_prefix, '_N_', num2str(this.N), '_', num2str(in_i), '.mat'];
                else
                    chro_name = [this.chro_prefix, '_', num2str(in_i), '.mat'];
                end
                % check if the chromosome file exists (applicable only to the chromosomes of the first round of repair)
                % if it exists, reuse it; otherwise, generate it
                if chroflag == 1 && exist(chro_name,'file') == 0 && repair_count == 0
                    error("The chromosome file does not exist or the chromosome name is incorrect!");
                    % chroflag = 0;
                    % continue;
                elseif chroflag == 1 && exist(chro_name,'file') && repair_count == 0
                    load(chro_name);
                    disp("Load the corresponding chromosome file successfully!")
                    repair_count = repair_count + 1;
                    rep_log{1, end+1} = [this.T_l, this.T_r, repair_count];
                elseif chroflag == 0 && exist(chro_name,'file')
                    error("The chromosome file exists, recheck the chroflag!");
                else
                    % if the nsga-ii algorithm is executed for the first time, save the chromosome as a .mat file
                    repair_count = repair_count + 1;
                    % update the repair log
                    rep_log{1, end+1} = [this.T_l, this.T_r, repair_count];
                    % if current repair reaches the repair budget, stop it
                    if repair_count > this.rep_budget
                        % % record the index of the input signals hard to repair and the strategy used for repair
                        % hard_pair = [in_i, this.st_id, this.rep_budget];
                        % failed_list = [failed_list; hard_pair];
                        traj_nn_in = {};
                        traj_nn_out = {};
                        new_rob = init_rob;
                        % if reaching the repair budget yields no solutions, then also save the chromosome set as a .mat file
                        % Use chro_name which already includes t_l_ratio and t_r_ratio if use_custom_search_range is true
                        % Clean function handles that capture AutoRepair object to reduce file size
                        if isstruct(chromosome) && isfield(chromosome, 'opt')
                            if isfield(chromosome.opt, 'objfun')
                                chromosome.opt.objfun = [];
                            end
                            if isfield(chromosome.opt, 'outputfuns')
                                chromosome.opt.outputfuns = {};
                            end
                            if isfield(chromosome.opt, 'initfun')
                                chromosome.opt.initfun = {};
                            end
                        end
                        save(chro_name, 'chromosome');
                        chroflag = 1;
                        return;
                    end

                    % define the lower bound and the upper bound of the patch_sig
                    oc_lb = [];
                    oc_ub = [];
                    % in case the dim of the control signal is greater than 1
                    % here, we manually set the lower bound and the upper bound to the subset of the range of the original control signal
                    for i = 1: numel(this.patch_range)
                        oc_lb = [oc_lb, repmat(this.patch_range{1,i}(:,1), [1, this.N])];
                        oc_ub = [oc_ub, repmat(this.patch_range{1,i}(:,2), [1, this.N])];
                    end
                    % set the bound of t_l and t_r as 0, to ensure smooth transitions of the patch at temporal boundaries.
                    oc_lb(1,1) = 0;
                    oc_lb(1,end) = 0;
                    oc_ub(1,1) = 0;
                    oc_ub(1,end) = 0;

                    % nsga_ii parameters
                    options = nsgaopt();                                     % create default options structure
                    options.popsize = this.pop_size;                         % populaion size
                    options.maxGen  = this.max_gen;                          % max generation
                    options.numObj = this.M;                                 % the number of the objectives
                    options.numVar = this.V;                                 % the number of the decision variables
                    options.numCons = 1;                                     % the number of the constraints
                    % Calculate search range bounds
                    if this.use_custom_search_range
                        % Use custom search range based on t_l_ratio and t_r_ratio
                        % Left endpoint: t_l_lb = opt_pos + t_l_ratio * (T_l - opt_pos), aligned to Ts
                        t_l_lb_raw = this.opt_pos + this.t_l_ratio * (this.T_l - this.opt_pos);
                        t_l_lb = round(t_l_lb_raw / this.Ts) * this.Ts;
                        % Right endpoint: t_r_ub = T_r - t_r_ratio * (T_r - T_l), aligned to Ts
                        t_r_ub_raw = this.T_r - this.t_r_ratio * (this.T_r - this.T_l);
                        t_r_ub = round(t_r_ub_raw / this.Ts) * this.Ts;
                        fprintf('[SearchRange] trace=%d, rep=%d, opt_pos=%.6g, T_l=%.6g, T_r=%.6g, t_l_lb=%.6g, t_r_ub=%.6g\n', ...
                            in_i, repair_count, this.opt_pos, this.T_l, this.T_r, t_l_lb, t_r_ub);
                        options.lb = [t_l_lb, t_l_lb, oc_lb];                   % the lower bound of x
                        options.ub = [this.T_l, t_r_ub, oc_ub];                 % the upper bound of x
                    else
                        % Use original search range (opt_pos to T_l/T_r)
                        options.lb = [this.opt_pos, this.opt_pos, oc_lb];        % the lower bound of x
                        options.ub = [this.T_l, this.T_r, oc_ub];                % the upper bound of x
                    end
                    options.objfun = @(x)this.repairObjFun(x,input_1,input_2);        % objective function handle
                    options.plotInterval = 5;                                % interval between two calls of "plotnsga".
                    options.outputfuns = {};                                 % disable output functions (prevents population.txt generation)
                    chromosome = nsga2(options);                             % begin the optimization!
                end

                % select qualified chromosomes
                cand_pool = [];

                for pop_i = 1:this.pop_size
                    % if there are more than three objectives, modify this part
                    obj1 = -chromosome.pops(this.max_gen,pop_i).obj(1,1);
                    obj2 = chromosome.pops(this.max_gen,pop_i).obj(1,2);
                    obj3 = chromosome.pops(this.max_gen,pop_i).obj(1,3);
                    % select the individuals with a negative rob (the rob of a 'good' individual should be a negative value)
                    if (obj1 > 0)
                        % index, rob, delta_t, L1-norm of the control signal patch
                        cand = [pop_i, obj1, obj2, obj3];
                        cand_pool = [cand_pool; cand];
                    end
                end

                [cand_r, ~] = size(cand_pool);

                % if there are no candidates
                if cand_r == 0
                    safe_patch_sig = zeros(1, this.sw_sz);
                    patch_sig = zeros(1, this.sw_sz);
                    continue;
                    % error('no solution satisfies that rob > 0');
                elseif chroflag == 0
                    % save the chromosome set as a .mat file
                    % Use chro_name which already includes t_l_ratio and t_r_ratio if use_custom_search_range is true
                    % Clean function handles that capture AutoRepair object to reduce file size
                    if isstruct(chromosome) && isfield(chromosome, 'opt')
                        if isfield(chromosome.opt, 'objfun')
                            chromosome.opt.objfun = [];
                        end
                        if isfield(chromosome.opt, 'outputfuns')
                            chromosome.opt.outputfuns = {};
                        end
                        if isfield(chromosome.opt, 'initfun')
                            chromosome.opt.initfun = {};
                        end
                    end
                    save(chro_name, 'chromosome');
                    chroflag = 1;
                end

                if this.st_id == 1
                    %% min_rob
                    [sort_cand_pool,~] = sortrows(cand_pool, 2);
                    % select the chromosome according to the rob
                    sel_chromosome = chromosome.pops(this.max_gen, sort_cand_pool(1,1));
                elseif this.st_id == 2
                    %% max_rob
                    [sort_cand_pool,~] = sortrows(cand_pool, 2, 'descend');
                    % select the chromosome according to the rob
                    sel_chromosome = chromosome.pops(this.max_gen, sort_cand_pool(1,1));
                elseif this.st_id == 3
                    %% min_change_time
                    [sort_cand_pool,~] = sortrows(cand_pool, 3);
                    % select the chromosome according to the delta_t
                    sel_chromosome = chromosome.pops(this.max_gen, sort_cand_pool(1,1));
                elseif this.st_id == 4
                    %% min_change_L1-norm
                    [sort_cand_pool,~] = sortrows(cand_pool, 4);
                    % select the chromosome according to the L1-norm
                    sel_chromosome = chromosome.pops(this.max_gen, sort_cand_pool(1,1));
                elseif this.st_id == 5
                    %% random
                    sel_pop_i = randi([1, cand_r]);
                    % select the chromosome randomly
                    sel_chromosome = chromosome.pops(this.max_gen, cand_pool(sel_pop_i,1));
                end

                % restore the control signal patch according to the selected chromosome
                t_l = sel_chromosome.var(1,1);
                t_r = sel_chromosome.var(1,2);
                t_l = roundn(t_l,0);
                t_r = roundn(t_r,0);

                % the left and right endpoint
                pos_l = int32(t_l/this.Ts)+1;
                pos_r = int32(t_r/this.Ts)+1;
                % patch signal (2 + N default)
                disc_patch = sel_chromosome.var(1,3:this.V);
                spacing = (t_r-t_l)/(this.N-1);
                t_control = t_l:spacing:t_r;
                v_control = disc_patch;

                % interplation, inp_patch_sig can not be set as a global variable
                if (t_r - t_l)/this.Ts >= 3
                    inp_patch_sig = interp1(t_control, v_control, t_l:this.Ts:t_r, 'makima');
                else
                    inp_patch_sig = mean(v_control) * ones(1, int32((t_r - t_l)/this.Ts) + 1);
                end

                % constrain the range of the control signal patch (here we only consider the case where the controller output dimension is 1)
                ub_i = inp_patch_sig(1,:) > this.patch_range{1,1}(1,2);
                inp_patch_sig(ub_i) = this.patch_range{1,1}(1,2);
                lb_i = inp_patch_sig(1,:) < this.patch_range{1,1}(1,1);
                inp_patch_sig(lb_i) = this.patch_range{1,1}(1,1);

                safe_patch_sig(1,pos_l:pos_r) = inp_patch_sig;
                patch_sig = zeros(1, this.sw_sz);

                % calculate the robustness value for signal diagnosis (based on the switch)
                % collect the input signals and the output signal of the nn controller
                % signal diagnosis 
                if strcmp(this.bm, 'uuv')
                    [new_rob, ~, this.T_l, this.T_r, traj_nn_in, traj_nn_out, ~, ~] = this.simMatlabUUV(input_1, input_2, 'benchmark/uuv/uuv_tanh_2_15_2x32_broken.yml', true, safe_patch_sig);
                elseif strcmp(this.bm, 'mc')
                    [new_rob, this.T_l, this.T_r, traj_nn_in, traj_nn_out, ~, ~] = this.simMatlabMC(input_1, input_2, 'benchmark/mc/mc_sig_2x16_broken.yml', true, safe_patch_sig);
                else
                    error('Check your benchmark!');
                end
               
                % recheck new robustness value
                if new_rob > 0
                    break;
                end

                % obtain the original control signal
                this.ocb_sig = traj_nn_out;
            end
        end
    end
end
