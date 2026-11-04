"""Generate the repaired-model evaluation scripts for AFC.

Each worker simulates through a reused fastRob session (util/sim: native sim()
in fast restart) instead of rebuilding Breach's interface for every test case.
Robustness values, metrics and saved variables are the same as with the
earlier Breach-based loop.
"""
import os
import sys
# path
addpath = []
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
fastrob_dir = os.path.join(repo_root, 'util', 'sim')
# dataset
D_run = []
D_train = []
D_test = []
D_repaired = []
# used to load neccessary param of model
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
				D_train.append(argu[0])
			elif arg == 'D_test':
				D_test.append(argu[0])
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
				in_span = '{'
				for idx in range(len(argu)-1):
					in_span = in_span + argu[idx]+','
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


def write_eval_loop(bm, s):
	"""Simulate one split (s = 'tr' or 'te') and compute its per-case metrics.

	The loop layout (worker i takes cases (i-1)*batch+1 .. i*batch) and the
	metric code are the same as in rep_eval_afc.py, so the saved sets line up
	case by case with the Breach-based evaluation.
	"""
	bm.write(s + '_loop_tic = tic;\n')
	bm.write('parfor i = 1:core_num\n')
	# One folder per worker process, kept for the whole job: the session built
	# in the first loop lives on in the second, and its model copy is in here.
	bm.write('\t worker_dir = fullfile(job_dir, sprintf(\'.rep_eval_worker_%s_%d\', eval_stamp, feature(\'getpid\')));\n')
	bm.write('\t if ~isfolder(worker_dir), mkdir(worker_dir); end\n')
	bm.write('\t old_dir = cd(worker_dir);\n')
	bm.write('\t ws_names = fieldnames(model_ws);\n')
	bm.write('\t for vi = 1:numel(ws_names)\n')
	bm.write('\t\t assignin(\'base\', ws_names{vi}, model_ws.(ws_names{vi}));\n')
	bm.write('\t end\n')
	bm.write('\t h = fastRobOpen(mdl_info, sim_conf);\n')
	bm.write('\t af_col = find(strcmp(h.sigNames, \'AF\'), 1);\n')
	bm.write('\t if isempty(af_col), error(\'rep_eval:noAF\', \'AF is not a signal of phi, so fastRob does not record it\'); end\n')
	bm.write('\t for subi = 1:' + s + '_batch\n')
	bm.write('\t\t case_idx = (i-1)*' + s + '_batch+subi;\n')
	bm.write('\t\t sys_inputs = cell(1, numel(in_name));\n')
	bm.write('\t\t for k = 1:numel(in_name)\n')
	bm.write('\t\t\t sys_inputs{k} = ' + s + '_in_cell{1,case_idx}{1,k}(1, 1:input_gen.cp(k));\n')
	bm.write('\t\t end\n')
	bm.write('\t\t [' + s + '_rob, ' + s + '_tout, ' + s + '_X] = fastRob(h, sys_inputs);\n')
	# Breach rethrows a Simulink error and the job dies; do the same rather than
	# letting fastRob's -1e6 into the statistics.
	bm.write('\t\t if isempty(' + s + '_tout)\n')
	bm.write('\t\t\t error(\'rep_eval:simFailed\', \'' + s + ' case %d: simulation failed\', case_idx);\n')
	bm.write('\t\t end\n')
	bm.write('\t\t if numel(' + s + '_tout) ~= numel(eval_timespan)\n')
	bm.write('\t\t\t error(\'rep_eval:timeGrid\', \'' + s + ' case %d: %d samples, expected %d\', case_idx, numel(' + s + '_tout), numel(eval_timespan));\n')
	bm.write('\t\t end\n')
	bm.write('\t\t disp([\'' + s + ' worker \', num2str(i), \' subi \', num2str(subi), \' rob = \', num2str(' + s + '_rob)]);\n')
	bm.write('\t\t ' + s + '_rob_set(i,subi) = ' + s + '_rob;\n')
	bm.write('\t\t ' + s + '_AF_sig = ' + s + '_X(:, af_col)\';\n')
	bm.write('\t\t ' + s + '_mu_sig = abs(ones(size(' + s + '_AF_sig)) - ' + s + '_AF_sig/14.7);\n')
	bm.write('\t\t steady_idx = find(' + s + '_mu_sig <= steady_const);\n')
	bm.write('\t\t ' + s + '_steady_set(i,subi) = numel(steady_idx)/numel(' + s + '_mu_sig);\n')
	bm.write('\t\t unsteady_count = 0;\n')
	bm.write('\t\t recovery_count = 0;\n')
	bm.write('\t\t unsteady_idx = find(' + s + '_mu_sig > steady_const);\n')
	bm.write('\t\t unsteady_interval = {};\n')
	bm.write('\t\t if numel(unsteady_idx) == 0 || numel(unsteady_idx) == 1\n')
	bm.write('\t\t\t unsteady_count = numel(unsteady_idx);\n')
	bm.write('\t\t\t recovery_count = numel(unsteady_idx);\n')
	bm.write('\t\t else\n')
	bm.write('\t\t\t inter = find(diff(unsteady_idx)~=1);\n')
	bm.write('\t\t\t if numel(inter) == 0\n')
	bm.write('\t\t\t\t unsteady_interval{1,1} = unsteady_idx;\n')
	bm.write('\t\t\t else\n')
	bm.write('\t\t\t\t for in_i = 1:length(inter) + 1\n')
	bm.write('\t\t\t\t\t if in_i == 1\n')
	bm.write('\t\t\t\t\t\t unsteady_interval{1,in_i} = unsteady_idx(1:inter(1));\n')
	bm.write('\t\t\t\t\t elseif in_i == length(inter) + 1\n')
	bm.write('\t\t\t\t\t\t unsteady_interval{1,in_i} = unsteady_idx(inter(in_i-1)+1:end);\n')
	bm.write('\t\t\t\t\t else\n')
	bm.write('\t\t\t\t\t\t unsteady_interval{1,in_i} = unsteady_idx(inter(in_i-1)+1:inter(in_i));\n')
	bm.write('\t\t\t\t\t end\n')
	bm.write('\t\t\t\t end\n')
	bm.write('\t\t\t end\n')
	bm.write('\t\t\t unsteady_count = numel(unsteady_interval);\n')
	bm.write('\t\t\t for un_i = 1: unsteady_count\n')
	bm.write('\t\t\t\t if numel(unsteady_interval{1, un_i}) < resilience_const\n')
	bm.write('\t\t\t\t recovery_count = recovery_count + 1;\n')
	bm.write('\t\t\t\t end\n')
	bm.write('\t\t\t end\n')
	bm.write('\t\t end\n')
	bm.write('\t\t ' + s + '_unsteady_set(i,subi) = unsteady_count;\n')
	bm.write('\t\t ' + s + '_recovery_set(i,subi) = recovery_count;\n')
	bm.write('\t\t ' + s + '_MAE_set(i,subi) = mean(' + s + '_mu_sig);\n')
	bm.write('\t\t ' + s + '_MAXERR_set(i,subi) = max(' + s + '_mu_sig);\n')
	bm.write('\t end\n')
	bm.write('\t cd(old_dir);\n')
	bm.write('end\n')
	bm.write(s + '_sim_time = toc(' + s + '_loop_tic);\n')
	for name in ['rob', 'steady', 'unsteady', 'recovery', 'MAE', 'MAXERR']:
		var = s + '_' + name + '_set'
		bm.write(var + ' = ' + var + '\';\n')
		bm.write(var + ' = ' + var + '(:);\n')


def write_cleanup(bm, indent):
	bm.write(indent + 'if exist(\'job_dir\', \'var\') && exist(\'eval_stamp\', \'var\')\n')
	bm.write(indent + '\tfor d = dir(fullfile(job_dir, sprintf(\'.rep_eval_worker_%s_*\', eval_stamp)))\'\n')
	bm.write(indent + '\t\tif d.isdir, rmdir(fullfile(d.folder, d.name), \'s\'); end\n')
	bm.write(indent + '\tend\n')
	bm.write(indent + 'end\n')
	bm.write(indent + 'if exist(\'sl_cache_dir\', \'var\')\n')
	bm.write(indent + '\ttry, Simulink.fileGenControl(\'reset\'); catch, end\n')
	bm.write(indent + '\tif isfolder(sl_cache_dir), rmdir(sl_cache_dir, \'s\'); end\n')
	bm.write(indent + 'end\n')


# script name
for mdl_i in range(len(model)):
	for phi_i in range(len(phi_str)):
		mdl = model[mdl_i]
		D_ru = D_run[mdl_i]
		D_rep = D_repaired[mdl_i]
		D_tr = D_train[mdl_i] if len(D_train) > 1 else D_train[0]
		D_te = D_test[mdl_i] if len(D_test) > 1 else D_test[0]
		property = phi_str[phi_i].split(';')
		# executable script name
		filename = str(mdl) + '_' + str(property[0]) + '_Rep_Eval'
		print(filename)
		param = '\n'.join(parameters)
		with open('scripts/'+filename,'w') as bm:
			bm.write('#!/bin/sh\n')
			bm.write('set -eu\n')
			bm.write('csv=$1\n')
			bm.write('matlab -nodesktop -nosplash <<EOF\n')
			bm.write('clear;\n')
			bm.write('close all;\n')
			bm.write('clc;\n')
			bm.write('bdclose(\'all\');\n')
			bm.write('tic;\n')
			# The outer try/catch is essential when MATLAB is driven through stdin:
			# without it, an error returns to the prompt and later save commands still run.
			bm.write('try\n')
			# addpath
			for ap in addpath:
				bm.write('addpath(genpath(\'' + ap + '\'));\n')
			bm.write('addpath(\'' + fastrob_dir.replace('\\', '/') + '\');\n')
			bm.write('bm = \'' + bench + '\';\n')
			# load dataset
			if D_run != '':
				bm.write('D_run = load(\'' + str(D_ru) + '\');\n')
			if D_train:
				bm.write('D_train = load(\'' + str(D_tr) + '\');\n')
			if D_test:
				bm.write('D_test = load(\'' + str(D_te) + '\');\n')
				bm.write('if ~isfield(D_test, \'te_in_cell\')\n')
				bm.write('\t D_test.te_in_cell = D_test.tr_in_cell;\n')
				bm.write('end\n')
				bm.write('if ~isfield(D_test, \'te_rob_set\')\n')
				bm.write('\t D_test.te_rob_set = D_test.tr_rob_set;\n')
				bm.write('end\n')
			if D_repaired:
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
			# The Breach-based script only sets inputs for UniStep; anything else
			# would not be a like-for-like comparison.
			bm.write('if ~strcmp(con_type, \'UniStep\'), error(\'rep_eval supports UniStep inputs only, got %s\', con_type); end\n')
			bm.write('phi_str = \'' + property[1] + '\';\n')
			bm.write('phi = STL_Formula(\'phi\',phi_str);\n')

			# prepare for parallel computing
			bm.write('core_num = ' +  core_num + ';\n')
			bm.write('mdl_path = which([mdl, \'.slx\']);\n')
			bm.write('if isempty(mdl_path), error(\'Model not found: %s.slx\', mdl); end\n')
			bm.write('job_dir = pwd;\n')
			bm.write('eval_stamp = datestr(now, \'yyyymmddHHMMSSFFF\');\n')
			bm.write('repo_root = \'' + repo_root.replace('\\', '/') + '\';\n')
			bm.write('sl_cache_dir = fullfile(repo_root, sprintf(\'rep_eval_slcache_%s\', eval_stamp));\n')
			bm.write('Simulink.fileGenControl(\'set\', \'CacheFolder\', sl_cache_dir, \'CodeGenFolder\', sl_cache_dir, \'createDir\', true);\n')
			bm.write('tr_num = numel(D_train.tr_in_cell);\n')
			bm.write('te_num = numel(D_test.te_in_cell);\n')
			bm.write('tr_batch = tr_num/core_num;\n')
			bm.write('te_batch = te_num/core_num;\n')
			bm.write('tr_rob_set = zeros(core_num, tr_batch);\n')
			bm.write('te_rob_set = zeros(core_num, te_batch);\n')
			bm.write('steady_const = 0.10;\n')
			bm.write('tr_steady_set = zeros(core_num, tr_batch);\n')
			bm.write('te_steady_set = zeros(core_num, te_batch);\n')
			bm.write('resilience_const = 10;\n')
			bm.write('tr_unsteady_set = zeros(core_num, tr_batch);\n')
			bm.write('te_unsteady_set = zeros(core_num, te_batch);\n')
			bm.write('tr_recovery_set = zeros(core_num, tr_batch);\n')
			bm.write('te_recovery_set = zeros(core_num, te_batch);\n')
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
			# InitBreach is a script. Running it directly in parfor can violate
			# workspace transparency, so initialize every worker before the loops.
			bm.write('init_futures = parfevalOnAll(pool, @evalin, 0, \'base\', \'InitBreach;\');\n')
			bm.write('fetchOutputs(init_futures);\n')
			bm.write('input_gen.type = con_type;\n')
			bm.write('input_gen.cp = T./cell2mat(in_span);\n')
			bm.write('eval_timespan = 0:Ts:T;\n')
			bm.write('load_system(mdl);\n')
			bm.write('mdl_var_list = Simulink.findVars(mdl, \'SearchMethod\', \'compiled\', \'SourceType\', \'base workspace\');\n')
			bm.write('model_ws = struct();\n')
			bm.write('for vi = 1:numel(mdl_var_list)\n')
			bm.write('\t vn = mdl_var_list(vi).Name;\n')
			bm.write('\t model_ws.(vn) = eval(vn);\n')
			bm.write('end\n')
			bm.write('close_system(mdl, 0);\n')

			# fastRob session description. No suspicious weights: the retrained
			# controller is evaluated as-is, so no weight is ever written.
			bm.write('mdl_info = struct(\'faultyMdl\', mdl, \'spsWeights\', zeros(0, 3), \'oriWeights\', {{}});\n')
			bm.write('sim_conf = struct(\'T\', T, \'Ts\', Ts, \'phi\', phi, \'in_span\', cell2mat(in_span), \'in_name\', {in_name});\n')

			## training dataset eval section
			write_eval_loop(bm, 'tr')

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
			bm.write('tr_mean_recovery = sum(tr_recovery_set)/sum(tr_unsteady_set);\n')
			bm.write('tr_mean_MAE = mean(tr_MAE_set);\n')
			bm.write('tr_mean_MAXERR = mean(tr_MAXERR_set);\n')

			## test dataset eval section
			write_eval_loop(bm, 'te')

			# Take every worker's model out of fast restart before the pool goes,
			# or MATLAB prompts on exit.
			bm.write('fastRobClose();\n')

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
			bm.write('te_mean_recovery = sum(te_recovery_set)/sum(te_unsteady_set);\n')
			bm.write('te_mean_MAE = mean(te_MAE_set);\n')
			bm.write('te_mean_MAXERR = mean(te_MAXERR_set);\n')
			bm.write('assert(numel(tr_rob_set) == tr_num && numel(te_rob_set) == te_num, ...\n')
			bm.write('\t \'Incomplete repeated-evaluation output; refusing to save.\');\n')
			bm.write('assert(all(isfinite(tr_rob_set)) && all(isfinite(te_rob_set)), ...\n')
			bm.write('\t \'Non-finite repeated-evaluation output; refusing to save.\');\n')
			bm.write('assert(any(tr_rob_set ~= 0) && any(te_rob_set ~= 0), ...\n')
			bm.write('\t \'All-zero repeated-evaluation output; refusing to save.\');\n')

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
			bm.write('save(eval_filename, \'tr_rob_set\', \'tr_neg_idx_bef\', \'tr_pos_idx_bef\', \'tr_neg_idx_aft\', \'tr_pos_idx_aft\', \'tr_neg_to_pos\', \'tr_pos_to_neg\', \'te_rob_set\', \'te_neg_idx_bef\', \'te_pos_idx_bef\', \'te_neg_idx_aft\', \'te_pos_idx_aft\', \'te_neg_to_pos\', \'te_pos_to_neg\', \'D_repaired_pos_idx\', \'D_repaired_neg_idx\', \'D_repaired_SR\', \'tr_SR\', \'te_SR\', \'tr_steady_set\', \'tr_unsteady_set\', \'tr_recovery_set\', \'tr_MAE_set\', \'tr_MAXERR_set\', \'tr_mean_steady\', \'tr_mean_recovery\', \'tr_mean_MAE\', \'tr_mean_MAXERR\', \'te_steady_set\', \'te_unsteady_set\', \'te_recovery_set\', \'te_MAE_set\', \'te_MAXERR_set\', \'te_mean_steady\', \'te_mean_recovery\', \'te_mean_MAE\', \'te_mean_MAXERR\');\n')
			bm.write('rep_eval_time = toc;\n')
			time_filename = '\'' + mdl + '_Rep_Eval_Time.mat' + '\''
			bm.write('time_filename = ' + time_filename + ';\n')
			bm.write('save(time_filename, \'rep_eval_time\', \'tr_sim_time\', \'te_sim_time\');\n')
			bm.write('fprintf(\'rep_eval_time = %.1f s (tr sims %.1f s, te sims %.1f s)\\n\', rep_eval_time, tr_sim_time, te_sim_time);\n')
			bm.write('bdclose(\'all\');\n')
			bm.write('pool = gcp(\'nocreate\');\n')
			bm.write('if ~isempty(pool), delete(pool); end\n')
			write_cleanup(bm, '')
			bm.write('catch ME\n')
			bm.write('\ttry, fastRobClose(); catch, end\n')
			bm.write('\ttry, bdclose(\'all\'); catch, end\n')
			bm.write('\tpool = gcp(\'nocreate\');\n')
			bm.write('\tif ~isempty(pool), delete(pool); end\n')
			write_cleanup(bm, '\t')
			bm.write('\tfprintf(2, \'%s\\n\', getReport(ME, \'extended\', \'hyperlinks\', \'off\'));\n')
			bm.write('\texit(1);\n')
			bm.write('end\n')
			bm.write('quit force\n')
			bm.write('EOF\n')
		os.chmod('scripts/' + filename, 0o755)
