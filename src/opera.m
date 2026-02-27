classdef opera < handle
    properties
        %% model 
        bm              % benchmark
        mdl_bef         % the model before repair
        mdl_rep         % the model used for repair
        D               % the dataset to be repaired
        T               % simulation time
        Ts              % sampling interval
        T_l             % the first instance when the robustness value turns negative
        T_r             % the first instance when the negative robustness dispears or the instance when the simulation ends
        %% the input signal of the system
        in_name         % the input signals of the system
        in_range        % the range of the input signals
        in_span         % the time span of the input signals
        %% the input signal of the controller
        icc_name        % the input constant of the controller
        ics_name        % the input signals of the controller
        ic_const        % the input constant value of the controller
        ic_sig          % the input signal values of the controller
        %% the output signal of the controller
        ocb_name        % the output signal of the controller before repair
        ocm_name        % the mixed output signal controlled by a plus block (patch_sig + a_ego_b)
        ocf_name        % the final output signal controlled by sw
        ocb_sig         % the value of the output signal of the controller before repair
        ocf_sig         % the value of the final output signal controlled by sw
        oc_range        % the range of the output signal
        oc_span         % the time span of the output signal
        %% specification parameters
        phi             % the specification in the form of STL
        phic            % the customized specification
        phic_str        % the string of the customized specification 
        interval_LR     % the interval specified in the specification (for intersection)
        %% multi-objective optimization parameters
        pop_size        % population size
        max_gen         % maximum number of the generation
        M               % the number of the objectives
        V               % the number of the decision variables
        N               % the number of the interpolation points
        use_custom_patch_points  % whether to use custom patch points setting (vary N/V) when use_custom_search_range is false
        amp             % the ampultute of the control signal patch
        patch_range     % the range of the control signal patch
        st_id           % strategy id
        opt_pos         % the customized start time for optimization (narrow down the search space)
        t_l_ratio       % ratio for t_l lower bound in [opt_pos, T_l] range (0-1), where t_l_lb = opt_pos + t_l_ratio * (T_l - opt_pos)
        t_r_ratio       % ratio for t_r upper bound in [T_l, T_r] range (0-1), where t_r_ub = T_r - t_r_ratio * (T_r - T_l)
        use_custom_search_range  % whether to use custom search range (t_l_ratio and t_r_ratio) or use default (opt_pos to T_l/T_r)
        chroflag        % exist chromosome files or not
        chro_prefix     % chromosome prefix  
        rep_budget      % repair budget
    end
    methods
        function this = opera(bm, mdl_bef, mdl_rep, D, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, ocb_name, ocm_name, ocf_name, oc_range, oc_span, phi, phic_str, interval_LR, pop_size, max_gen, M, V, N, use_custom_patch_points, amp, st_id, opt_pos, use_custom_search_range, t_l_ratio, t_r_ratio, chro_prefix, rep_budget)

            this.bm = bm;
            this.mdl_bef = mdl_bef;
            this.mdl_rep = mdl_rep;
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

            this.phi = phi;
            this.phic_str = phic_str;
            this.interval_LR = interval_LR;

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
                this.patch_range{1,end+1} = this.amp * this.oc_range{1,i};
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
        function [rob, Br] = calRob(this, mdl, in_sig, phi)
            % Given a model and an input signal, calRob function returns the robustness and Br, according to the specification.
            %
            % Inputs:
            %   mdl: a simulink model
            %   in_sig: an external input signal of the system
            %   phi: specification
            % Outputs:
            %   rob: robustness value
            %   Br: an instance of the class BreachSimulinkSystem
            
            Br = BreachSimulinkSystem(mdl);
            Br.Sys.tspan = 0:this.Ts:this.T;

            if length(unique(this.in_span)) == 1
                con_type = 'UniStep';
                input_gen.cp = this.T/this.in_span(1,1);
            else
                con_type = 'VarStep';
                input_gen.cp = this.T./this.in_span;
            end
            input_gen.type = con_type;
            Br.SetInputGen(input_gen);

            if strcmp(con_type, 'UniStep')
                for i = 1:numel(this.in_name)
                    for cpi = 0:input_gen.cp - 1
                        eval(['Br.SetParam({''',this.in_name{1,i},'_u',num2str(cpi),'''}, in_sig{1,i}(1,cpi+1));']);
                    end
                end
            elseif strcmp(con_type, 'VarStep')
                for i = 1:numel(this.in_name)
                    for cpi = 0: input_gen.cp(1,i) - 1
                        eval(['Br.SetParam({''',this.in_name{1,i},'_u',num2str(cpi),'''}, in_sig{1,i}(1,cpi+1));']);
                        if cpi ~= input_gen.cp(1,i) - 1
                            eval(['Br.SetParam({''',this.in_name{1,i},'_dt',num2str(cpi),'''},this.in_span(1,i));']);
                        end
                    end
                end
            end

            Br.Sim(0:this.Ts:this.T);
            rob = Br.CheckSpec(phi);
        end

        function [T_l, T_r] = signalDiagnosis(this, Br)
            % signalDiagnosis function performs signal diagnosis to obtain T_l and T_r.
            %
            % Inputs:
            %   Br: an instance of the class BreachSimulinkSystem
            % Outputs:
            %   T_l: the first instance when the robustness value turns negative
            %   T_r: the first instance when the negative robustness dispears or the instance when the simulation ends
            
            % declare a variable as a global variable
            global sw_sz;

            % scan the interval specified in the specification
            scan_interval = zeros(1, sw_sz);
            % negetive interval stores the violation information of current system execution. if the value at a certain timestamp is 1,
            % it means the system's behavior violates the specification at this timestamp.
            neg_interval = zeros(1, sw_sz);

            % in scan_interval, the timestamps which belong to the interval specified in the specification are set to 1 (for intersection)
            for i = 1: numel(this.interval_LR)
                LR = this.interval_LR{1,i};
                LR = LR./this.Ts;
                scan_interval(1, LR(1,1)+1:LR(1,2)+1) = 1;
            end
            
            if strcmp(this.bm, 'ACC')
                % phi_str_acc = 'alw_[0,50]((d_rel[t] - 1.4 * v_ego[t] >= 10) and v_ego[t] <= 30.1)';
                negl_interval = sigMatch(Br, "d_rel") - 1.4 * sigMatch(Br, "v_ego") < 10;
                negr_interval = sigMatch(Br, "v_ego") > 30.1;
                neg_idx = find(negl_interval + negr_interval > 0);
                neg_interval(1, neg_idx) = 1;
            elseif strcmp(this.bm, 'AFC')
                % phi_str_afc = 'alw_[0,30](AF[t] < 1.15*14.7 and AF[t] > 0.85*14.7)';
                af = sigMatch(Br, "AF");
                mu = abs(af - 14.7)/14.7;
                neg_interval(mu >= 0.15) = 1;
            elseif strcmp(this.bm, 'WT')
                % phi_str_wt = phi_str = 'alw_[4,4.9](abs(h_error[t]) < 0.86) and alw_[9,9.9](abs(h_error[t]) < 0.86) and alw_[14,14.9](abs(h_error[t]) < 0.86)';
                h_error = abs(sigMatch(Br, "h_error"));
                neg_interval(h_error > 0.86) = 1;
            end
                
            % obtain the intersection of neg_interval and scan_interval
            int_interval = neg_interval .* scan_interval;

            if ~ismember(1,int_interval)
                error('signal diagnosis fails due to the empty int_interval!');
            end

            % obtain the T_l and T_r
            neg_idx = find(int_interval == 1);
            negl_idx = neg_idx(1,1);
            
            for i = negl_idx:sw_sz
                if int_interval(1, i) == 1
                    negr_idx = i;
                else
                    break;
                end
            end
            
            T_l = (negl_idx - 1) * this.Ts;
            T_r = (negr_idx - 1) * this.Ts;
        end

        function [y, cons] = repairObjFun(this, x, in_sig)
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
            
            % declare global variables
            global sw_sz patch_sig;

            % initialize y and cons
            y = [];
            cons = [];

            % calculate the constraint violations
            c = x(2) - x(1);
            if c < 0
                cons(1) = abs(c);
            end

            % round to the nearest tenth
            t_l = roundn(x(1),-1);
            t_r = roundn(x(2),-1);

            % a hard constraint: t_l < t_r
            if t_l >= t_r || c < 0
                y = [inf,inf,inf];
                disp(y);
                % reset patch_sig
                patch_sig = zeros(1, sw_sz);
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
            pos_l = int32(t_l/this.Ts) + 1;
            pos_r = int32(t_r/this.Ts) + 1;
            
            patch_sig = zeros(1, sw_sz);
            patch_sig(1, pos_l:pos_r) = inp_patch_sig;

            % objective function 1
            [rob, ~] = this.calRob(this.mdl_rep, in_sig, this.phic);
            % objective function 2
            dist = t_r - t_l;
            % objective function 3
            sig_diff = sum(abs(inp_patch_sig));

            % output
            y = [-rob, dist, sig_diff];
            disp(y);

            % reset patch_sig
            patch_sig = zeros(1, sw_sz);
        end

        function [ic_sig_val, oc_sig_val, new_rob, rep_log] = sigRepair(this, in_sig, chroflag, in_i)
            % sigRepair function can synthesize the control signal for a given external input signal of the system.
            %
            % Inputs:
            %   in_sig: a given input signal of the system
            %   chroflag: exist chromosome or not (applicable only to chromosomes of the first round of repair). If it exists, reuse it; otherwise, generate it
            %   in_i: the idx of in_sig, used for visualizing the repair progress
            % Outputs:
            %   ic_sig_val: the repaired input signal of the nn controller
            %   oc_sig_val: the repaired output signal of the nn controller
            %   new_rob: new robustness value after repair
            %   rep_log: repair log, e.g. {[this.T_l, this.T_r, repair_count]} 
            
            % declare global variables
            global sw_sz sw patch_sig safe_patch_sig failed_list;
            % initialize sw, patch_sig and safe_patch_sig
            sw = ones(1, sw_sz);
            patch_sig = zeros(1, sw_sz);
            safe_patch_sig = zeros(1, sw_sz);
            % initialize the repair log
            rep_log = {};

            % recheck the robustness value to prevent inconsistency of the robustness values caused by the dataset or Breach
            [init_rob, Br_rep] = this.calRob(this.mdl_rep, in_sig, this.phi);
            % simulation error (due to different versions of breach or flexiable time step)
            if init_rob > 0
                ic_sig_val = sigMatch(Br_rep, this.ics_name);
                oc_sig_val = sigMatch(Br_rep, this.ocf_name);
                new_rob = init_rob;
                rep_log{1,end+1} = [0,0,0];
                disp('The robustness value has been corrected!');
                return;
            elseif abs(init_rob - this.D.tr_rob_set(in_i,1)) >= 0.01
                error('The calculated robustness value does not match the original robustness value.');
            end

            % signal diagnosis
            [this.T_l, this.T_r] = this.signalDiagnosis(Br_rep);
            % customize the spec based on this.T_l and this.T_r
            this.phic = phiReplace(this.phic_str, this.interval_LR, this.T_l, this.T_r);
            % get the control signal of the nn controller to be repaired
            this.ocb_sig = sigMatch(Br_rep, this.ocb_name);
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
                        % record the index of the input signals hard to repair and the strategy used for repair
                        hard_pair = [in_i, this.st_id, this.rep_budget];
                        failed_list = [failed_list; hard_pair];
                        ic_sig_val = {};
                        oc_sig_val = {};
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

                    % customize this.opt_pos, for example, the specification of WT is 
                    % phi = 'alw_[4,4.9](abs(h_error[t]) < 0.86) and alw_[9,9.9](abs(h_error[t]) < 0.86) and alw_[14,14.9](abs(h_error[t]) < 0.86)'
                    % in this case, the interval where the violation occurred is either [4,4.9], [9,9.9], or [14,14.9].
                    % since both the input signals and the output signal are periodic signals, it means the if the violation
                    % occurs in the interval [14,14.9], we should specify the search interval as [10,15].
                    % if there is no explicit range, then this range will be determined through dataset.
                    
                    % this section need to be improved!!!!!!!!!!!!
                    if strcmp(this.bm, 'WT')
                        for int_i = 1:numel(this.interval_LR)
                            if this.T_l >= this.interval_LR{1,int_i}(1,1) && this.T_r <= this.interval_LR{1,int_i}(1,2)
                                break;
                            end
                        end
                        
                        if int_i == 1
                            this.opt_pos = 0;
                            this.oc_range = {[1.75,30]};
                        elseif int_i == 2
                            this.opt_pos = 5;
                            this.oc_range = {[-20,20]};
                        elseif int_i == 3
                            this.opt_pos = 10;
                            this.oc_range = {[-20,20]};
                        end
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
                    options.objfun = @(x)this.repairObjFun(x,in_sig);        % objective function handle
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
                    % reset sw, repair from the first violation episode
                    sw = ones(1, sw_sz);
                    safe_patch_sig = zeros(1, sw_sz);
                    patch_sig = zeros(1, sw_sz);
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
                t_l = roundn(t_l,-1);
                t_r = roundn(t_r,-1);

                % the left and right endpoint
                pos_l = int32(t_l/this.Ts)+1;
                pos_r = int32(t_r/this.Ts)+1;
                % patch signal (2 + N default)
                disc_patch = sel_chromosome.var(1,3:this.V);

                % interplation, inp_patch_sig can not be set as a global variable
                if (t_r - t_l)/this.Ts >= 3
                    spacing = (t_r-t_l)/(this.N-1);
                    inp_patch_sig = interp1(t_l:spacing:t_r, disc_patch, t_l:this.Ts:t_r, 'makima');
                else
                    inp_patch_sig = mean(disc_patch) * ones(1, int32(t_r - t_l)/this.Ts + 1);
                end

                % constrain the range of the control signal patch (here we only consider the case where the controller output dimension is 1)
                ub_i = inp_patch_sig(1,:) > this.patch_range{1,1}(1,2);
                inp_patch_sig(ub_i) = this.patch_range{1,1}(1,2);
                lb_i = inp_patch_sig(1,:) < this.patch_range{1,1}(1,1);
                inp_patch_sig(lb_i) = this.patch_range{1,1}(1,1);

                % modify sw
                sw(1,1:pos_r) = 0;
                safe_patch_sig(1,pos_l:pos_r) = inp_patch_sig;
                patch_sig = zeros(1, sw_sz);

                % calculate the robustness value for signal diagnosis (based on the switch)
                [new_rob, Br_rep] = this.calRob(this.mdl_rep, in_sig, this.phi);

                % collect the input signals and the output signal of the nn controller
                ic_sig_val = sigMatch(Br_rep, this.ics_name);
                oc_sig_val = sigMatch(Br_rep, this.ocf_name);

                % recheck new robustness value
                if new_rob > 0
                    break;
                end

                % signal diagnosis
                [this.T_l, this.T_r] = this.signalDiagnosis(Br_rep);
                % customize the spec based on this.T_l and this.T_r
                this.phic = phiReplace(this.phic_str, this.interval_LR, this.T_l, this.T_r);
                % obtain the original control signal
                this.ocb_sig = sigMatch(Br_rep, this.ocb_name);
            end
        end
    end
end
