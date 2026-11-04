import os
import sys
# path
addpath = []
fastrob_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'util', 'sim')
# dataset
D_run = []
D_train = ''
D_test = ''
D_repaired = []
# used to load neccessary param of the model 
is_nor = ''
# model parameters info
bench = ''
model = []
parameters = []
# other configuration
in_name = []
in_range = []
in_span = []
con_type = ''
phi_str = []
# parallel computing
core_num = ''
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
			elif arg == 'D_repaired':
				D_repaired.append(argu[0])
			elif arg == 'is_nor':
				is_nor = argu[0]
			elif arg == 'bench':
				bench = argu[0]
			elif arg == 'model':
				model.append(argu[0])
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
			elif arg == 'core_num':
				core_num = argu[0]
			else:
				continue
			if linenum == 0:
				status = 0



# script name
for mdl_i in range(len(model)):
	for phi_i in range(len(phi_str)):
		mdl = model[mdl_i]
		D_ru = D_run[mdl_i]
		D_rep = D_repaired[mdl_i]
		property = phi_str[phi_i].split(';')
		# executable script name
		filename = str(mdl) + '_' + str(property[0]) + '_Eval'
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
			bm.write('tic;\n')
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
			if D_repaired != '':
				bm.write('D_repaired = load(\'' + str(D_rep) + '\');\n')
			# model parameters info
			bm.write('mdl = \''+ mdl + '\';\n')
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
			# prepare for parallel computing
			bm.write('core_num = ' +  core_num + ';\n')
			bm.write('mdl_path = which([mdl, \'.slx\']);\n')
			bm.write('tr_num = numel(D_train.tr_in_cell);\n')
			bm.write('te_num = numel(D_test.te_in_cell);\n')
			bm.write('tr_batch = tr_num/core_num;\n')
			bm.write('te_batch = te_num/core_num;\n')
			# one row, one mdl
			bm.write('tr_rob_set = zeros(core_num, tr_batch);\n')
			bm.write('te_rob_set = zeros(core_num, te_batch);\n')
			bm.write('steady_const = 0.5;\n')
			bm.write('tr_steady_set = zeros(core_num, tr_batch);\n')
			bm.write('te_steady_set = zeros(core_num, te_batch);\n')
			bm.write('tr_MAE_set = zeros(core_num, tr_batch);\n')
			bm.write('te_MAE_set = zeros(core_num, te_batch);\n')
			bm.write('tr_MAXERR_set = zeros(core_num, tr_batch);\n')
			bm.write('te_MAXERR_set = zeros(core_num, te_batch);\n')
			
			bm.write('tr_in_cell = D_train.tr_in_cell;\n')
			bm.write('te_in_cell = D_test.te_in_cell;\n')
			
			# initialize a parallel pool
			bm.write('pool = gcp(\'nocreate\');\n')
			bm.write('if ~isempty(pool)\n')
			bm.write('\t delete(pool);\n')
			bm.write('end\n')
			bm.write('pool = parpool(core_num);\n')

			bm.write('InitBreach;\n')
			bm.write('input_gen.type = con_type;\n')
			bm.write('input_gen.cp = T./cell2mat(in_span);\n')

			# fastRob: each worker simulates through one reused session (util/sim)
			bm.write('addpath(\'' + fastrob_dir.replace('\\', '/') + '\');\n')
			bm.write('if ~strcmp(con_type, \'UniStep\'), error(\'rep_eval supports UniStep inputs only, got %s\', con_type); end\n')
			bm.write('job_dir = pwd;\n')
			bm.write('eval_stamp = datestr(now, \'yyyymmddHHMMSSFFF\');\n')
			bm.write('eval_timespan = 0:Ts:T;\n')
			bm.write('init_futures = parfevalOnAll(pool, @evalin, 0, \'base\', \'InitBreach;\');\n')
			bm.write('fetchOutputs(init_futures);\n')
			bm.write('load_system(mdl);\n')
			bm.write('mdl_var_list = Simulink.findVars(mdl, \'SearchMethod\', \'compiled\', \'SourceType\', \'base workspace\');\n')
			bm.write('model_ws = struct();\n')
			bm.write('for vi = 1:numel(mdl_var_list)\n')
			bm.write('\t vn = mdl_var_list(vi).Name;\n')
			bm.write('\t model_ws.(vn) = eval(vn);\n')
			bm.write('end\n')
			bm.write('close_system(mdl, 0);\n')
			bm.write('mdl_info = struct(\'faultyMdl\', mdl, \'spsWeights\', zeros(0, 3), \'oriWeights\', {{}});\n')
			bm.write('sim_conf = struct(\'T\', T, \'Ts\', Ts, \'phi\', phi, \'in_span\', cell2mat(in_span), \'in_name\', {in_name}, \'extraSigs\', {{\'h_error\'}});\n')
			## training dataset eval section
			bm.write('parfor i = 1:core_num\n')
			bm.write('\t worker_dir = fullfile(job_dir, sprintf(\'.rep_eval_worker_%s_%d\', eval_stamp, feature(\'getpid\')));\n')
			bm.write('\t if ~isfolder(worker_dir), mkdir(worker_dir); end\n')
			bm.write('\t old_dir = cd(worker_dir);\n')
			bm.write('\t ws_names = fieldnames(model_ws);\n')
			bm.write('\t for vi = 1:numel(ws_names)\n')
			bm.write('\t\t assignin(\'base\', ws_names{vi}, model_ws.(ws_names{vi}));\n')
			bm.write('\t end\n')
			bm.write('\t h = fastRobOpen(mdl_info, sim_conf);\n')
			bm.write('\t for subi = 1:tr_batch\n')
			bm.write('\t\t case_idx = (i-1)*tr_batch+subi;\n')
			bm.write('\t\t sys_inputs = cell(1, numel(in_name));\n')
			bm.write('\t\t for k = 1:numel(in_name)\n')
			bm.write('\t\t\t sys_inputs{k} = tr_in_cell{1,case_idx}{1,k}(1, 1:input_gen.cp(k));\n')
			bm.write('\t\t end\n')
			bm.write('\t\t [tr_rob, tr_tout, tr_X] = fastRob(h, sys_inputs);\n')
			bm.write('\t\t if isempty(tr_tout)\n')
			bm.write('\t\t\t error(\'rep_eval:simFailed\', \'tr case %d: simulation failed\', case_idx);\n')
			bm.write('\t\t end\n')
			bm.write('\t\t if numel(tr_tout) ~= numel(eval_timespan)\n')
			bm.write('\t\t\t error(\'rep_eval:timeGrid\', \'tr case %d: %d samples, expected %d\', case_idx, numel(tr_tout), numel(eval_timespan));\n')
			bm.write('\t\t end\n')
			bm.write('\t\t tr_rob_set(i,subi) = tr_rob;\n')
			bm.write('\t\t tr_error_sig = tr_X(:, strcmp(h.sigNames, \'h_error\'))\';\n')
			bm.write('\t\t tr_sel_error_sig = [tr_error_sig(1, 41:50), tr_error_sig(1, 91:100), tr_error_sig(1, 141:150)];\n')
			bm.write('\t\t tr_sel_error_sig = abs(tr_sel_error_sig);\n')
			bm.write('\t\t tr_steady_count = numel(find(tr_sel_error_sig < steady_const));\n')
			bm.write('\t\t tr_steady_rate = tr_steady_count/numel(tr_sel_error_sig);\n')
			bm.write('\t\t tr_steady_set(i,subi) = tr_steady_rate;\n')
			bm.write('\t\t tr_MAE = mean(tr_sel_error_sig);\n')
			bm.write('\t\t tr_MAE_set(i,subi) = tr_MAE;\n')
			bm.write('\t\t tr_MAXERR = max(tr_sel_error_sig);\n')
			bm.write('\t\t tr_MAXERR_set(i,subi) = tr_MAXERR;\n')
			bm.write('\t end\n')
			bm.write('\t cd(old_dir);\n')
			bm.write('end\n')
			# reconstruct
			bm.write('tr_rob_set = tr_rob_set\';\n')
			bm.write('tr_rob_set = tr_rob_set(:);\n')
			bm.write('tr_steady_set = tr_steady_set\';\n')
			bm.write('tr_steady_set = tr_steady_set(:);\n')
			bm.write('tr_MAE_set = tr_MAE_set\';\n')
			bm.write('tr_MAE_set = tr_MAE_set(:);\n')
			bm.write('tr_MAXERR_set = tr_MAXERR_set\';\n')
			bm.write('tr_MAXERR_set = tr_MAXERR_set(:);\n')
			# handle the training dataset collected before 
			bm.write('tr_neg_idx_bef = find(D_train.tr_rob_set < 0);\n')
			bm.write('tr_pos_idx_bef = find(D_train.tr_rob_set > 0);\n')
			# handle the training dataset after repair
			bm.write('tr_neg_idx_aft = find(tr_rob_set < 0);\n')
			bm.write('tr_pos_idx_aft = find(tr_rob_set > 0);\n')
			# perform statistical analyses
			bm.write('tr_alw_pos = intersect(tr_pos_idx_bef, tr_pos_idx_aft);\n')
			bm.write('tr_alw_neg = intersect(tr_neg_idx_bef, tr_neg_idx_aft);\n')
			bm.write('tr_neg_to_pos = setdiff(tr_pos_idx_aft, tr_alw_pos);\n')
			bm.write('tr_pos_to_neg = setdiff(tr_neg_idx_aft, tr_alw_neg);\n')
			bm.write('tr_mean_steady = mean(tr_steady_set);\n')
			bm.write('tr_mean_MAE = mean(tr_MAE_set);\n')
			bm.write('tr_mean_MAXERR = mean(tr_MAXERR_set);\n')
			
			## test dataset eval section
			bm.write('parfor i = 1:core_num\n')
			bm.write('\t worker_dir = fullfile(job_dir, sprintf(\'.rep_eval_worker_%s_%d\', eval_stamp, feature(\'getpid\')));\n')
			bm.write('\t if ~isfolder(worker_dir), mkdir(worker_dir); end\n')
			bm.write('\t old_dir = cd(worker_dir);\n')
			bm.write('\t ws_names = fieldnames(model_ws);\n')
			bm.write('\t for vi = 1:numel(ws_names)\n')
			bm.write('\t\t assignin(\'base\', ws_names{vi}, model_ws.(ws_names{vi}));\n')
			bm.write('\t end\n')
			bm.write('\t h = fastRobOpen(mdl_info, sim_conf);\n')
			bm.write('\t for subi = 1:te_batch\n')
			bm.write('\t\t case_idx = (i-1)*te_batch+subi;\n')
			bm.write('\t\t sys_inputs = cell(1, numel(in_name));\n')
			bm.write('\t\t for k = 1:numel(in_name)\n')
			bm.write('\t\t\t sys_inputs{k} = te_in_cell{1,case_idx}{1,k}(1, 1:input_gen.cp(k));\n')
			bm.write('\t\t end\n')
			bm.write('\t\t [te_rob, te_tout, te_X] = fastRob(h, sys_inputs);\n')
			bm.write('\t\t if isempty(te_tout)\n')
			bm.write('\t\t\t error(\'rep_eval:simFailed\', \'te case %d: simulation failed\', case_idx);\n')
			bm.write('\t\t end\n')
			bm.write('\t\t if numel(te_tout) ~= numel(eval_timespan)\n')
			bm.write('\t\t\t error(\'rep_eval:timeGrid\', \'te case %d: %d samples, expected %d\', case_idx, numel(te_tout), numel(eval_timespan));\n')
			bm.write('\t\t end\n')
			bm.write('\t\t te_rob_set(i,subi) = te_rob;\n')
			bm.write('\t\t te_error_sig = te_X(:, strcmp(h.sigNames, \'h_error\'))\';\n')
			bm.write('\t\t te_sel_error_sig = [te_error_sig(1, 41:50), te_error_sig(1, 91:100), te_error_sig(1, 141:150)];\n')
			bm.write('\t\t te_sel_error_sig = abs(te_sel_error_sig);\n')
			bm.write('\t\t te_steady_count = numel(find(te_sel_error_sig < steady_const));\n')
			bm.write('\t\t te_steady_rate = te_steady_count/numel(te_sel_error_sig);\n')
			bm.write('\t\t te_steady_set(i,subi) = te_steady_rate;\n')
			bm.write('\t\t te_MAE = mean(te_sel_error_sig);\n')
			bm.write('\t\t te_MAE_set(i,subi) = te_MAE;\n')
			bm.write('\t\t te_MAXERR = max(te_sel_error_sig);\n')
			bm.write('\t\t te_MAXERR_set(i,subi) = te_MAXERR;\n')
			bm.write('\t end\n')
			bm.write('\t cd(old_dir);\n')
			bm.write('end\n')
			# reconstruct
			bm.write('te_rob_set = te_rob_set\';\n')
			bm.write('te_rob_set = te_rob_set(:);\n')
			bm.write('te_steady_set = te_steady_set\';\n')
			bm.write('te_steady_set = te_steady_set(:);\n')
			bm.write('te_MAE_set = te_MAE_set\';\n')
			bm.write('te_MAE_set = te_MAE_set(:);\n')
			bm.write('te_MAXERR_set = te_MAXERR_set\';\n')
			bm.write('te_MAXERR_set = te_MAXERR_set(:);\n')
			# handle the test dataset collected before 
			bm.write('te_neg_idx_bef = find(D_test.te_rob_set < 0);\n')
			bm.write('te_pos_idx_bef = find(D_test.te_rob_set > 0);\n')
			# handle the test dataset after repair
			bm.write('te_neg_idx_aft = find(te_rob_set < 0);\n')
			bm.write('te_pos_idx_aft = find(te_rob_set > 0);\n')
			# perform statistical analyses
			bm.write('te_alw_pos = intersect(te_pos_idx_bef, te_pos_idx_aft);\n')
			bm.write('te_alw_neg = intersect(te_neg_idx_bef, te_neg_idx_aft);\n')
			bm.write('te_neg_to_pos = setdiff(te_pos_idx_aft, te_alw_pos);\n')
			bm.write('te_pos_to_neg = setdiff(te_neg_idx_aft, te_alw_neg);\n')
			bm.write('te_mean_steady = mean(te_steady_set);\n')
			bm.write('te_mean_MAE = mean(te_MAE_set);\n')
			bm.write('te_mean_MAXERR = mean(te_MAXERR_set);\n')
			# take the workers' models out of fast restart and drop their folders
			bm.write('fastRobClose();\n')
			bm.write('for d = dir(fullfile(job_dir, sprintf(\'.rep_eval_worker_%s_*\', eval_stamp)))\'\n')
			bm.write('\t if d.isdir, rmdir(fullfile(d.folder, d.name), \'s\'); end\n')
			bm.write('end\n')
			# success rate of repaired dataset
			bm.write('D_repaired_pos_idx = find(D_repaired.tr_rob_set > 0);\n')
			bm.write('D_repaired_neg_idx = find(D_repaired.tr_rob_set < 0);\n')
			bm.write('D_repaired_SR = (numel(tr_neg_idx_bef) - numel(D_repaired_neg_idx))/numel(tr_neg_idx_bef);\n')
			# success rate on training dataset 
			# tr_SR is the number of satisfied cases (divide by the number of cases for the rate)
			bm.write('tr_SR = numel(find(tr_rob_set > 0));\n')
			# success rate on test dataset
			# te_SR is the number of satisfied cases (divide by the number of cases for the rate)
			bm.write('te_SR = numel(find(te_rob_set > 0));\n')
			## save evaluation result as .mat file 
			eval_filename = '\'' + mdl + '_Rep_Eval_Result.mat' + '\''
			bm.write('eval_filename = ' + eval_filename + ';\n')
			bm.write('save(eval_filename, \'tr_rob_set\', \'tr_neg_idx_bef\', \'tr_pos_idx_bef\', \'tr_neg_idx_aft\', \'tr_pos_idx_aft\', \'tr_neg_to_pos\', \'tr_pos_to_neg\', \'te_rob_set\', \'te_neg_idx_bef\', \'te_pos_idx_bef\', \'te_neg_idx_aft\', \'te_pos_idx_aft\', \'te_neg_to_pos\', \'te_pos_to_neg\', \'D_repaired_pos_idx\', \'D_repaired_neg_idx\', \'D_repaired_SR\', \'tr_SR\', \'te_SR\', \'tr_steady_set\', \'tr_MAE_set\', \'tr_MAXERR_set\', \'tr_mean_steady\', \'tr_mean_MAE\', \'tr_mean_MAXERR\', \'te_steady_set\', \'te_MAE_set\', \'te_MAXERR_set\', \'te_mean_steady\', \'te_mean_MAE\', \'te_mean_MAXERR\');\n')
			bm.write('rep_eval_time = toc;\n')
			time_filename = '\'' + mdl + '_Rep_Eval_Time.mat' + '\''
			bm.write('time_filename = ' + time_filename + ';\n')
			bm.write('save(time_filename, \'rep_eval_time\');\n')
			bm.write('delete(fullfile([\'breach/Ext/ModelsData/\', \'_breach*\']));\n')
			bm.write('delete(fullfile([\'.slx\']));\n')
			bm.write('delete(fullfile([\'.slx.r202*\']));\n')
			bm.write('delete(fullfile([\'_breach.slxc\']));\n')
			bm.write('quit force\n')
			bm.write('EOF\n')