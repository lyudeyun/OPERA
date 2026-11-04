function  sig_val = sigMatch(Br, sig_list)
% sigMatch.m return the sequences of specific signals, given a list of
% signal names.
% Inputs:
%   Br: BreachSimulinkSystem of the Simulink model
%   sig_list: the list of signals to be obtained
% Outputs:
%   sig_val: the sequences of specific signals
tmp_sig_list = Br.GetSignalList();
tmp_sig_val = Br.P.traj{1,1}.X;

sig_val = [];
for i = 1: numel(sig_list)
    for ti = 1: numel(tmp_sig_list)
        if strcmp(sig_list(i), tmp_sig_list(ti))
            sig_val(i,:) = tmp_sig_val(ti,:);
        end
    end
end
end