function generateDataSet(mdl, D_size, D_name, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, oc_name, phi)
% generateDataSet.m returns the training dataset.
% Besides, it also returns a sequence of input signals that
% violate the specification.
%
% Inputs:
%   mdl: simulink model
%   D_size: the required size of the training dataset
%   D_name: the naming rule of dataset
% Outputs:
%   D: the training dataset

Br = BreachSimulinkSystem(mdl);
Br.Sys.tspan = 0:Ts:T;

% training dataset
for tr_idx = 1:D_size
    disp('train');
    disp(tr_idx);
    [in_cell, ic_cell, oc_cell, rob] = oneSim(Br, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, oc_name, phi);
    if tr_idx == 1
        tr_in_cell{1,1} = in_cell;
        tr_ic_cell = ic_cell;
        tr_oc_cell = oc_cell;
        tr_rob_set = rob;
        continue;
    else
        tr_in_cell{1,tr_idx} = in_cell;
        tr_ic_cell = catsamples(tr_ic_cell, ic_cell, 'pad');
        tr_oc_cell = catsamples(tr_oc_cell, oc_cell, 'pad');
    end
    tr_rob_set = [tr_rob_set; rob];
end
save(D_name, 'tr_in_cell', 'tr_ic_cell', 'tr_oc_cell', 'tr_rob_set');
end


function [in_cell, ic_cell, oc_cell, rob] = oneSim(Br, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, oc_name, phi)
% oneSim.m returns the input and output signals of
% controller in the form of a cell. Besides, the robustness also will be recorded.
%
% Inputs:
%   Br - BreachSimulinkSystem of the Simulink model
% Outputs:
%   in_cell - external input signals
%   ic_cell - input signals of controller
%   oc_cell - output signals of controller
%   rob: the robustness

in_span = cell2mat(in_span);
if isscalar(unique(in_span))
    con_type = 'UniStep';
    input_gen.type = con_type;
    input_gen.cp = T / in_span(1);
else
    con_type = 'VarStep';
    input_gen.type = con_type;
    input_gen.cp = T ./ in_span;
end

Br.SetInputGen(input_gen);

% initialize in, ic, oc
in_cell = {};
ic_mat = [];
oc_mat = [];

% randomly generate input signals
for i = 1:numel(in_name)
    if strcmp(con_type, 'UniStep')
        eval([in_name{1,i},'_set = unifrnd(in_range{1,i}(1,1), in_range{1,i}(1,2), 1, input_gen.cp);']);
    else
        eval([in_name{1,i},'_set = unifrnd(in_range{1,i}(1,1), in_range{1,i}(1,2), 1, input_gen.cp(i));']);
    end
    eval(['in_cell{1,i} =', in_name{1,i},'_set;']);
end

if strcmp(con_type, 'UniStep')
    for i = 1:numel(in_name)
        for cpi = 0:input_gen.cp - 1
            eval(['Br.SetParam({''', in_name{1,i},'_u',num2str(cpi),'''},', in_name{1,i},'_set(1,cpi+1));']);
        end
    end
    % multiple input signals
elseif strcmp(con_type, 'VarStep')
    for i = 1:numel(in_name)
        for cpi = 0: input_gen.cp(1,i) - 1
            eval(['Br.SetParam({''', in_name{1,i},'_u',num2str(cpi),'''},', in_name{1,i},'_set(1,cpi+1));']);
            if cpi ~= input_gen.cp(1,i) - 1
                eval(['Br.SetParam({''', in_name{1,i},'_dt',num2str(cpi),'''}, in_span(1,i));']);
            end
        end
    end
end

Br.Sim(0:Ts:T);

ic_sig_val = sigMatch(Br, ics_name);
% trad model
oc_sig_val = sigMatch(Br, oc_name);

for i = 1:numel(icc_name)
    eval([icc_name{1,i}, '_set = ic_const{1,i} * ones(1, T/Ts+1);']);
end

% add input constants of controller
for i = 1: numel(icc_name)
    eval(['ic_mat = [ic_mat; ', icc_name{1,i}, '_set];']);
end

ic_mat = [ic_mat; ic_sig_val];
ic_cell = num2cell(ic_mat,1);

oc_mat = [oc_mat; oc_sig_val];
oc_cell = num2cell(oc_mat,1);

rob = Br.CheckSpec(phi);
end

