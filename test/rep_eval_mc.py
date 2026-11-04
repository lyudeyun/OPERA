"""Generate the repaired-model evaluation scripts for MC.

Like rep_eval_{acc,afc,wt}.py, one script per model, which saves
<model>_Rep_Eval_Result.mat and <model>_Rep_Eval_Time.mat in the current folder.
MC is not a Simulink model: the scripts call src/eval/MC_rep_eval.m, which
simulates the retrained network with benchmark/mc/simMatlabMC_Rep.m on the
training and test sets and computes safety (tr_/te_rob_set, _SR as the number
of satisfied cases, _neg_to_pos, _pos_to_neg) and the MC metrics (reach, energy,
smooth: _set per case, _mean_ over the set; *_bef for the original controller).
"""
import os
import sys
# path
addpath = []
# dataset
D_train = []
D_test = []
# model parameters info
bench = ''
model = []
# other configuration
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
			elif arg == 'D_train':
				D_train.append(argu[0])
			elif arg == 'D_test':
				D_test.append(argu[0])
			elif arg == 'bench':
				bench = argu[0]
			elif arg == 'model':
				model.append(argu[0])
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
		D_tr = D_train[mdl_i] if len(D_train) > 1 else D_train[0]
		D_te = D_test[mdl_i] if len(D_test) > 1 else D_test[0]
		property = phi_str[phi_i].split(';')
		# the evaluated model is named like the other benchmarks' repaired models: the network file without "config"
		mdl_name = mdl.replace('_config_', '_')
		# executable script name
		filename = mdl_name + '_' + str(property[0]) + '_Rep_Eval'
		print(filename)
		with open('scripts/'+filename,'w') as bm:
			bm.write('#!/bin/sh\n')
			bm.write('set -eu\n')
			bm.write('csv=$1\n')
			bm.write('matlab -nodesktop -nosplash <<EOF\n')
			bm.write('clear;\n')
			bm.write('close all;\n')
			bm.write('clc;\n')
			bm.write('tic;\n')
			# The outer try/catch is essential when MATLAB is driven through stdin:
			# without it, an error returns to the prompt and later save commands still run.
			bm.write('try\n')
			# addpath
			for ap in addpath:
				bm.write('addpath(genpath(\'' + ap + '\'));\n')
			bm.write('bm = \'' + bench + '\';\n')
			bm.write('core_num = ' + core_num + ';\n')
			bm.write('eval_filename = \'' + mdl_name + '_Rep_Eval_Result.mat\';\n')
			bm.write('res = MC_rep_eval(\'' + mdl + '.mat\', eval_filename, core_num, \'' + D_tr + '\', \'' + D_te + '\');\n')
			bm.write('rep_eval_time = toc;\n')
			bm.write('time_filename = \'' + mdl_name + '_Rep_Eval_Time.mat\';\n')
			bm.write('save(time_filename, \'rep_eval_time\');\n')
			bm.write('fprintf(\'rep_eval_time = %.1f s\\\\n\', rep_eval_time);\n')
			bm.write('pool = gcp(\'nocreate\');\n')
			bm.write('if ~isempty(pool), delete(pool); end\n')
			bm.write('catch ME\n')
			bm.write('\tpool = gcp(\'nocreate\');\n')
			bm.write('\tif ~isempty(pool), delete(pool); end\n')
			bm.write('\tfprintf(2, \'%s\\\\n\', getReport(ME, \'extended\', \'hyperlinks\', \'off\'));\n')
			bm.write('\texit(1);\n')
			bm.write('end\n')
			bm.write('quit force\n')
			bm.write('EOF\n')
		os.chmod('scripts/' + filename, 0o755)
