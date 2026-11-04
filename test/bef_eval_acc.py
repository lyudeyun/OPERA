import os
import sys
# path
addpath = []
fastrob_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'util', 'sim')
# dataset
D_run = []
D_train = ''
D_test = ''
# used to load neccessary param of model 
is_nor = ''
# model parameters info
bench = ''
model_bef = []
parameters = []
# other configuration
in_name = []
in_range = []
in_span = []
con_type = ''
phi_str = []
# script parameters
status = 0
arg = ''
linenum = 0

with open(sys.argv[1],'r') as conf:
	for line in conf.readlines():
		argu = line.strip().split()
		if status == 0:
			status = 1
			arg = argu[0]
			linenum = int(argu[1])
		elif status == 1:
			linenum = linenum - 1
			if arg == 'addpath':
				addpath.append(argu[0])
			elif arg == 'D_run':
				D_run.append(argu[0])
			elif arg == 'D_train':
				D_train = argu[0]
			elif arg == 'D_test':
				D_test = argu[0]
			elif arg == 'is_nor':
				is_nor = argu[0]
			elif arg == 'bench':
				bench = argu[0]
			elif arg == 'model_bef':
				model_bef.append(argu[0])
			elif arg == 'parameters':
				parameters.append(argu[0])
			elif arg == 'in_name':
				in_name.append(argu[0])
			elif arg == 'in_range':
				in_range.append([float(argu[0]),float(argu[1])])
			elif arg == 'in_span':
				# print(len(argu))
				in_span = '{'
				for idx in range(len(argu)-1):
					in_span = in_span + argu[idx]+','
					# print(argu[idx])
				in_span = in_span + argu[len(argu)-1] + '}'
			elif arg == 'con_type':
				con_type = argu[0]
			elif arg == 'phi_str':
				complete_phi = argu[0]+';'+argu[1]
				for a in argu[2:]:
					complete_phi = complete_phi + ' ' + a
				phi_str.append(complete_phi)
			else:
				continue
			if linenum == 0:
				status = 0
# script name
for mdl_i in range(len(model_bef)):
	for phi_i in range(len(phi_str)):
		mdl_bef = model_bef[mdl_i]
		D_ru = D_run[mdl_i]
		property = phi_str[phi_i].split(';')
		# executable script name
		filename = str(mdl_bef) + '_' + str(property[0]) + '_Bef_Eval'
		print(filename)
		param = '\n'.join(parameters)
		with open('scripts/'+filename,'w') as bm:
			bm.write('#!/bin/sh\n')
			bm.write('csv=$1\n')
			bm.write('matlab -nodesktop -nosplash <<EOF\n')
			bm.write('clear;\n')
			bm.write('close all;\n')
			bm.write('clc;\n')
			bm.write('bdclose(\'all\');\n')
			# addpath
			for ap in addpath:
				bm.write('addpath(genpath(\'' + ap + '\'));\n')
			bm.write('bm = \'' + bench + '\';\n')
			# load dataset
			if D_run != '':
				bm.write('D_run = load(\'' + str(D_ru) + '\');\n')
			if D_train != '':
				bm.write('D_train = load(\'' + D_train + '\');\n')
			if D_test != '':
				bm.write('D_test = load(\'' + D_test + '\');\n')
			# model parameters info
			bm.write('mdl_bef = \''+ mdl_bef + '\';\n')
			bm.write(param + '\n')
			bm.write('is_nor = ' + is_nor + ';\n')
			bm.write('if is_nor == 1\n')
			bm.write('\tx_gain = D_run.ps_x.gain;\n')
			bm.write('\tx_gain = diag(x_gain);\n')
			bm.write('\tx_offset = D_run.ps_x.xoffset;\n')
			bm.write('\ty_gain = D_run.ps_y.gain;\n')
			bm.write('\ty_offset = D_run.ps_y.xoffset;\n')
			bm.write('end\n')

			# other configuration
			bm.write('in_name = {\'' + in_name[0] + '\'')
			for inname in in_name[1:]:
				bm.write(',')
				bm.write('\'' + inname + '\'')
			bm.write('};\n')

			bm.write('in_range = {[' + str(in_range[0][0]) + ' ' + str(in_range[0][1]) + ']')
			for ir in in_range[1:]:
				bm.write(',[' + str(ir[0]) + ' ' + str(ir[1]) + ']')
			bm.write('};\n')

			bm.write('in_span = ' + in_span + ';\n')
			bm.write('con_type = \'' + con_type  + '\';\n')
			bm.write('phi_str = \'' + property[1] + '\';\n')
			bm.write('phi = STL_Formula(\'phi\',phi_str);\n')

			bm.write('tr_rob_set = zeros(10000,1);\n')
			bm.write('te_rob_set = zeros(1000,1);\n')
			bm.write('tr_x_ego_set = zeros(1,10000);\n')
			bm.write('te_x_ego_set = zeros(1,1000);\n')
			bm.write('steady_const = 2;\n')
			bm.write('tr_steady_num_set = zeros(1,10000);\n')
			bm.write('te_steady_num_set = zeros(1,1000);\n')
			bm.write('tr_delta_v_sum_set = zeros(1,10000);\n')
			bm.write('tr_delta_v_time_set = zeros(1,10000);\n')
			bm.write('te_delta_v_sum_set = zeros(1,1000);\n')
			bm.write('te_delta_v_time_set = zeros(1,1000);\n')
			bm.write('delta_a_ego_const = 0.1;\n')
			bm.write('tr_comfortable_set = zeros(1,10000);\n')
			bm.write('te_comfortable_set = zeros(1,1000);\n')
			bm.write('resilience_const = 10;\n')
			bm.write('tr_unsteady_num_set = zeros(1,10000);\n')
			bm.write('te_unsteady_num_set = zeros(1,1000);\n')
			bm.write('tr_recovery_num_set = zeros(1,10000);\n')
			bm.write('te_recovery_num_set = zeros(1,1000);\n')

			bm.write('InitBreach;\n')
			# fastRob: one simulation session (util/sim) reused for every test case
			bm.write('addpath(\'' + fastrob_dir.replace('\\', '/') + '\');\n')
			bm.write('if ~strcmp(con_type, \'UniStep\'), error(\'bef_eval supports UniStep inputs only, got %s\', con_type); end\n')
			bm.write('input_gen.type = con_type;\n')
			bm.write('input_gen.cp = T./cell2mat(in_span);\n')
			bm.write('eval_timespan = 0:Ts:T;\n')
			bm.write('job_dir = pwd;\n')
			bm.write('work_dir = fullfile(job_dir, sprintf(\'.bef_eval_%s_%d\', datestr(now, \'yyyymmddHHMMSSFFF\'), feature(\'getpid\')));\n')
			bm.write('mkdir(work_dir);\n')
			bm.write('cd(work_dir);\n')
			bm.write('mdl_info = struct(\'faultyMdl\', mdl_bef, \'spsWeights\', zeros(0, 3), \'oriWeights\', {{}});\n')
			bm.write('sim_conf = struct(\'T\', T, \'Ts\', Ts, \'phi\', phi, \'in_span\', cell2mat(in_span), \'in_name\', {in_name}, \'extraSigs\', {{\'x_ego\', \'d_rel\', \'v_ego\', \'a_ego\'}});\n')
			bm.write('h = fastRobOpen(mdl_info, sim_conf);\n')
			## training dataset eval section
			bm.write('for i = 1:numel(D_train.tr_in_cell)\n')
			bm.write('\tsys_inputs = cell(1, numel(in_name));\n')
			bm.write('\tfor k = 1:numel(in_name)\n')
			bm.write('\t\tsys_inputs{k} = D_train.tr_in_cell{1,i}{1,k}(1, 1:input_gen.cp(k));\n')
			bm.write('\tend\n')
			bm.write('\t[tr_rob, tr_tout, tr_X] = fastRob(h, sys_inputs);\n')
			bm.write('\tif isempty(tr_tout)\n')
			bm.write('\t\terror(\'bef_eval:simFailed\', \'tr case %d: simulation failed\', i);\n')
			bm.write('\tend\n')
			bm.write('\tif numel(tr_tout) ~= numel(eval_timespan)\n')
			bm.write('\t\terror(\'bef_eval:timeGrid\', \'tr case %d: %d samples, expected %d\', i, numel(tr_tout), numel(eval_timespan));\n')
			bm.write('\tend\n')
			bm.write('\ttr_rob_set(i,1) = tr_rob;\n')

			bm.write('\tx_ego_sig = tr_X(:, strcmp(h.sigNames, \'x_ego\'))\';\n')
			bm.write('\ttr_x_ego_set(1,i) = x_ego_sig(1,end);\n')
			bm.write('\tsteady_state = tr_X(:, strcmp(h.sigNames, \'d_rel\'))\' - 1.4 * tr_X(:, strcmp(h.sigNames, \'v_ego\'))\' - 10 - steady_const;\n')
			bm.write('\tsteady_idx = find(steady_state > 0);\n')
			bm.write('\ttr_steady_num_set(1,i) = numel(steady_idx)/501;\n')
			bm.write('\tsafety_state = tr_X(:, strcmp(h.sigNames, \'d_rel\'))\' - 1.4 * tr_X(:, strcmp(h.sigNames, \'v_ego\'))\' - 10;\n')
			bm.write('\tif numel(find(safety_state > 0)) == 0\n')
			bm.write('\t\tdisp(i);\n')
			bm.write('\t\tdisp(\'error occurs\');\n')
			bm.write('\tend\n')
			bm.write('\tsafety_idx = find(safety_state > 0);\n')
			bm.write('\tv_ego_sig = tr_X(:, strcmp(h.sigNames, \'v_ego\'))\';\n')
			bm.write('\tdelta_v_sum = sum(abs(v_ego_sig(safety_idx) - v_set));\n')
			bm.write('\ttr_delta_v_sum_set(1,i) = delta_v_sum;\n')
			bm.write('\ttr_delta_v_time_set(1,i) = numel(safety_idx);\n')
			bm.write('\ta_ego_sig = tr_X(:, strcmp(h.sigNames, \'a_ego\'))\';\n')
			bm.write('\tdelta_a_ego =  abs(diff(a_ego_sig));\n')
			bm.write('\tcomfortable_idx = find(delta_a_ego < delta_a_ego_const);\n')
			bm.write('\ttr_comfortable_set(1,i) = numel(comfortable_idx)/500;\n')
			bm.write('\tunsteady_idx = find(steady_state < 0);\n')
			bm.write('\tunsteady_num = 0;\n')
			bm.write('\trecovery_num = 0;\n')
			bm.write('\tunsteady_interval = {};\n')
			bm.write('\tif numel(unsteady_idx) == 0 || numel(unsteady_idx) == 1\n')
			bm.write('\t\tunsteady_num = numel(unsteady_idx);\n')
			bm.write('\t\trecovery_num = numel(unsteady_idx);\n')
			bm.write('\telse\n')
			bm.write('\t\tinter = find(diff(unsteady_idx)~=1);\n')
			bm.write('\t\tif numel(inter) == 0\n')
			bm.write('\t\t\tunsteady_interval{1,1} = unsteady_idx;\n')
			bm.write('\t\telse\n')
			bm.write('\t\t\tfor in_i = 1:length(inter)+1\n')
			bm.write('\t\t\t\tif in_i == 1\n')
			bm.write('\t\t\t\t\tunsteady_interval{1,1} = unsteady_idx(1:inter(1));\n')
			bm.write('\t\t\t\telseif in_i == length(inter) + 1\n')
			bm.write('\t\t\t\t\tunsteady_interval{1,in_i} = unsteady_idx(inter(in_i - 1) + 1:end);\n')
			bm.write('\t\t\t\telse\n')
			bm.write('\t\t\t\t\tunsteady_interval{1,in_i} = unsteady_idx(inter(in_i - 1) + 1:inter(in_i));\n')
			bm.write('\t\t\t\tend\n')
			bm.write('\t\t\tend\n')
			bm.write('\t\tend\n')
			bm.write('\t\tunsteady_num = numel(unsteady_interval);\n')
			bm.write('\t\tfor un_i = 1: unsteady_num\n')
			bm.write('\t\t\tif numel(unsteady_interval{1, un_i}) < resilience_const\n')
			bm.write('\t\t\t\trecovery_num = recovery_num + 1;\n')
			bm.write('\t\t\tend\n')
			bm.write('\t\tend\n')
			bm.write('\tend\n')
			bm.write('\ttr_unsteady_num_set(1,i) = unsteady_num;\n')
			bm.write('\ttr_recovery_num_set(1,i) = recovery_num;\n')
			bm.write('end\n')
			
			bm.write('tr_mean_x_ego = mean(tr_x_ego_set);\n')
			bm.write('tr_mean_steady = mean(tr_steady_num_set);\n')
			bm.write('tr_mean_delta_v = sum(tr_delta_v_sum_set)/sum(tr_delta_v_time_set);\n')
			bm.write('tr_mean_comfortable = mean(tr_comfortable_set);\n')
			bm.write('tr_mean_resilience = sum(tr_recovery_num_set)/sum(tr_unsteady_num_set);\n')

			## test dataset eval section
			bm.write('for i = 1:numel(D_test.te_in_cell)\n')
			bm.write('\tsys_inputs = cell(1, numel(in_name));\n')
			bm.write('\tfor k = 1:numel(in_name)\n')
			bm.write('\t\tsys_inputs{k} = D_test.te_in_cell{1,i}{1,k}(1, 1:input_gen.cp(k));\n')
			bm.write('\tend\n')
			bm.write('\t[te_rob, te_tout, te_X] = fastRob(h, sys_inputs);\n')
			bm.write('\tif isempty(te_tout)\n')
			bm.write('\t\terror(\'bef_eval:simFailed\', \'te case %d: simulation failed\', i);\n')
			bm.write('\tend\n')
			bm.write('\tif numel(te_tout) ~= numel(eval_timespan)\n')
			bm.write('\t\terror(\'bef_eval:timeGrid\', \'te case %d: %d samples, expected %d\', i, numel(te_tout), numel(eval_timespan));\n')
			bm.write('\tend\n')
			bm.write('\tte_rob_set(i,1) = te_rob;\n')

			bm.write('\tx_ego_sig = te_X(:, strcmp(h.sigNames, \'x_ego\'))\';\n')
			bm.write('\tte_x_ego_set(1,i) = x_ego_sig(1,end);\n')
			bm.write('\tsteady_state = te_X(:, strcmp(h.sigNames, \'d_rel\'))\' - 1.4 * te_X(:, strcmp(h.sigNames, \'v_ego\'))\' - 10 - steady_const;\n')
			bm.write('\tsteady_idx = find(steady_state > 0);\n')
			bm.write('\tte_steady_num_set(1,i) = numel(steady_idx)/501;\n')
			bm.write('\tsafety_state = te_X(:, strcmp(h.sigNames, \'d_rel\'))\' - 1.4 * te_X(:, strcmp(h.sigNames, \'v_ego\'))\' - 10;\n')
			bm.write('\tif numel(find(safety_state > 0)) == 0\n')
			bm.write('\t\tdisp(i);\n')
			bm.write('\t\tdisp(\'error occurs\');\n')
			bm.write('\tend\n')
			bm.write('\tsafety_idx = find(safety_state > 0);\n')
			bm.write('\tv_ego_sig = te_X(:, strcmp(h.sigNames, \'v_ego\'))\';\n')
			bm.write('\tdelta_v_sum = sum(abs(v_ego_sig(safety_idx) - v_set));\n')
			bm.write('\tte_delta_v_sum_set(1,i) = delta_v_sum;\n')
			bm.write('\tte_delta_v_time_set(1,i) = numel(safety_idx);\n')
			bm.write('\ta_ego_sig = te_X(:, strcmp(h.sigNames, \'a_ego\'))\';\n')
			bm.write('\tdelta_a_ego =  abs(diff(a_ego_sig));\n')
			bm.write('\tcomfortable_idx = find(delta_a_ego < delta_a_ego_const);\n')
			bm.write('\tte_comfortable_set(1,i) = numel(comfortable_idx)/500;\n')
			bm.write('\tunsteady_idx = find(steady_state < 0);\n')
			bm.write('\tunsteady_num = 0;\n')
			bm.write('\trecovery_num = 0;\n')
			bm.write('\tunsteady_interval = {};\n')
			bm.write('\tif numel(unsteady_idx) == 0 || numel(unsteady_idx) == 1\n')
			bm.write('\t\tunsteady_num = numel(unsteady_idx);\n')
			bm.write('\t\trecovery_num = numel(unsteady_idx);\n')
			bm.write('\telse\n')
			bm.write('\t\tinter = find(diff(unsteady_idx)~=1);\n')
			bm.write('\t\tif numel(inter) == 0\n')
			bm.write('\t\t\tunsteady_interval{1,1} = unsteady_idx;\n')
			bm.write('\t\telse\n')
			bm.write('\t\t\tfor in_i = 1:length(inter)+1\n')
			bm.write('\t\t\t\tif in_i == 1\n')
			bm.write('\t\t\t\t\tunsteady_interval{1,1} = unsteady_idx(1:inter(1));\n')
			bm.write('\t\t\t\telseif in_i == length(inter) + 1\n')
			bm.write('\t\t\t\t\tunsteady_interval{1,in_i} = unsteady_idx(inter(in_i - 1) + 1:end);\n')
			bm.write('\t\t\t\telse\n')
			bm.write('\t\t\t\t\tunsteady_interval{1,in_i} = unsteady_idx(inter(in_i - 1) + 1:inter(in_i));\n')
			bm.write('\t\t\t\tend\n')
			bm.write('\t\t\tend\n')
			bm.write('\t\tend\n')
			bm.write('\t\tunsteady_num = numel(unsteady_interval);\n')
			bm.write('\t\tfor un_i = 1: unsteady_num\n')
			bm.write('\t\t\tif numel(unsteady_interval{1, un_i}) < resilience_const\n')
			bm.write('\t\t\t\trecovery_num = recovery_num + 1;\n')
			bm.write('\t\t\tend\n')
			bm.write('\t\tend\n')
			bm.write('\tend\n')
			bm.write('\tte_unsteady_num_set(1,i) = unsteady_num;\n')
			bm.write('\tte_recovery_num_set(1,i) = recovery_num;\n')

			bm.write('end\n')
			
			bm.write('te_mean_x_ego = mean(te_x_ego_set);\n')
			bm.write('te_mean_steady = mean(te_steady_num_set);\n')
			bm.write('te_mean_delta_v = sum(te_delta_v_sum_set)/sum(te_delta_v_time_set);\n')
			bm.write('te_mean_comfortable = mean(te_comfortable_set);\n')
			bm.write('te_mean_resilience = sum(te_recovery_num_set)/sum(te_unsteady_num_set);\n')

			# take the model out of fast restart and drop the session's folder
			bm.write('fastRobClose();\n')
			bm.write('cd(job_dir);\n')
			bm.write('bdclose(\'all\');\n')
			bm.write('rmdir(work_dir, \'s\');\n')
			## save evaluation result as .mat file 
			eval_filename = '\'' + mdl_bef + '_Bef_Eval_Result.mat' + '\''
			bm.write('eval_filename = ' + eval_filename + ';\n')
			bm.write('save(eval_filename, \'tr_rob_set\', \'te_rob_set\', \'tr_x_ego_set\', \'tr_steady_num_set\', \'tr_delta_v_sum_set\', \'tr_delta_v_time_set\', \'tr_comfortable_set\', \'tr_unsteady_num_set\', \'tr_recovery_num_set\', \'tr_mean_x_ego\', \'tr_mean_steady\', \'tr_mean_delta_v\', \'tr_mean_comfortable\', \'tr_mean_resilience\', \'te_x_ego_set\', \'te_steady_num_set\', \'te_delta_v_sum_set\', \'te_delta_v_time_set\', \'te_comfortable_set\', \'te_unsteady_num_set\', \'te_recovery_num_set\', \'te_mean_x_ego\', \'te_mean_steady\', \'te_mean_delta_v\', \'te_mean_comfortable\', \'te_mean_resilience\');\n')
			bm.write('quit force\n')
			bm.write('EOF\n')