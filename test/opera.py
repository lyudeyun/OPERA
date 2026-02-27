import sys
# path
addpath = []
# the dataset for training this underperforming nn controllers, containing the necessary normalization parameters to run mdl_rep
dataset = ''
# dataset_tbr
dataset_tbr = ''
D_size = ''
is_nor = ''
# model parameters info
bench = ''
model_bef = ''
model_rep = ''
parameters = []
# other configuration
in_name = []
in_range = []
in_span = []
icc_name = []
ic_const = []
ics_name = []
ocb_name = []
ocm_name = []
ocf_name = []
oc_range = []
oc_span = []
phi_str = []
phic_str = []
interval_LR = []
# hyperparameters for multi-objective optimization 
pop_size = ''
max_gen = ''
M = ''
V = ''
N = ''
use_custom_patch_points = '0'  # Default: false (do not add N/V suffix to naming)
amp = ''
rep_budget = ''
st_id = ''
chro_prefix = ''
chroflag = []
opt_pos = []
t_l_ratio = ''
t_r_ratio = ''
use_custom_search_range = '0'  # Default: false (use original search range)
start_idx = ''
end_idx = ''
batch = '1'  # Default: 1 (generate single script)
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
			elif arg == 'dataset':
				dataset = argu[0]
			elif arg == 'dataset_tbr':
				dataset_tbr = argu[0]
			elif arg == 'chro_prefix':
				chro_prefix = argu[0]
			elif arg == 'D_size':
				D_size = argu[0]
			elif arg == 'is_nor':
				is_nor = argu[0]
			elif arg == 'bench':
				bench = argu[0]
			elif arg == 'model_bef':
				model_bef = argu[0]
			elif arg == 'model_rep':
				model_rep = argu[0]
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
			elif arg == 'icc_name':
				icc_name.append(argu[0])
			elif arg == 'ic_const':
				ic_const.append(argu[0])
			elif arg == 'ics_name':
				ics_name.append(argu[0])
			elif arg == 'ocb_name':
				ocb_name.append(argu[0])
			elif arg == 'ocm_name':
				ocm_name.append(argu[0])
			elif arg == 'ocf_name':
				ocf_name.append(argu[0])
			elif arg == 'oc_range':
				oc_range.append([float(argu[0]),float(argu[1])])
			elif arg == 'oc_span':
				oc_span = '{'
				for idx in range(len(argu)-1):
					oc_span = oc_span + argu[idx]+','
				oc_span = oc_span + argu[len(argu)-1] + '}'
			elif arg == 'phi_str':
				complete_phi = argu[0]+';'+argu[1]
				for a in argu[2:]:
					complete_phi = complete_phi + ' ' + a
				phi_str.append(complete_phi)
			elif arg == 'phic_str':
				complete_phic = argu[0]+';'+argu[1]
				for a in argu[2:]:
					complete_phic = complete_phic + ' ' + a
				phic_str.append(complete_phic)
			elif arg == 'interval_LR':
				interval_LR.append([float(argu[0]),float(argu[1])])
			elif arg == 'pop_size':
				pop_size = argu[0]
			elif arg == 'max_gen':
				max_gen = argu[0]
			elif arg == 'M':
				M = argu[0]
			elif arg == 'V':
				V = argu[0]
			elif arg == 'N':
				N = argu[0]
			elif arg == 'use_custom_patch_points':
				use_custom_patch_points = argu[0]
			elif arg == 'amp':
				amp = argu[0]
			elif arg == 'rep_budget':
				rep_budget = argu[0]
			elif arg == 'st_id':
				st_id = argu[0]
			elif arg == 'chroflag':
				chroflag = argu[0]
			elif arg == 'opt_pos':
				opt_pos = argu[0]
			elif arg == 't_l_ratio':
				t_l_ratio = argu[0]
			elif arg == 't_r_ratio':
				t_r_ratio = argu[0]
			elif arg == 'use_custom_search_range':
				use_custom_search_range = argu[0]
			elif arg == 'start_idx':
				start_idx = argu[0]
			elif arg == 'end_idx':
				end_idx = argu[0]
			elif arg == 'batch':
				batch = argu[0]
			else:
				continue
			if linenum == 0:
				status = 0

for phi_i in range(len(phi_str)):
	property = phi_str[phi_i].split(';')
	propertyc = phic_str[phi_i].split(';')
	
	# Calculate batch ranges
	num_batch = int(batch)
	start = int(start_idx)
	end = int(end_idx)
	total_traces = end - start + 1
	batch_size = total_traces // num_batch
	
	# Generate a script for each batch
	for batch_i in range(num_batch):
		# Calculate start and end for this batch
		batch_start = start + batch_i * batch_size
		if batch_i == num_batch - 1:
			# Last batch takes any remainder
			batch_end = end
		else:
			batch_end = batch_start + batch_size - 1
		
		batch_start_str = str(batch_start)
		batch_end_str = str(batch_end)
		
		# script name
		# Generate filename based on whether custom search range or custom patch points is used
		if use_custom_search_range == '1':
			filename = dataset_tbr[:-4] + '_St_'+ st_id + '_t_l_ratio_' + t_l_ratio + '_t_r_ratio_' + t_r_ratio + '_Start_' + batch_start_str + '_End_' + batch_end_str + '_Repair'
		elif use_custom_patch_points == '1':
			filename = dataset_tbr[:-4] + '_St_'+ st_id + '_N_' + N + '_Start_' + batch_start_str + '_End_' + batch_end_str + '_Repair'
		else:
			filename = dataset_tbr[:-4] + '_St_'+ st_id + '_Start_' + batch_start_str + '_End_' + batch_end_str + '_Repair'
		print(filename)
		logname = '\'./output/' + filename + '.log\''
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
			# load the dataset for training this underperforming nn controllers, containing the necessary normalization parameters to run mdl_rep
			if dataset != '':
				bm.write('D_run = load(\'' + dataset + '\');\n')
			# load dataset_tbr
			if dataset_tbr != '':
				bm.write('D = load(\'' + dataset_tbr + '\');\n')
				bm.write('D_size = ' + D_size + ';\n')
			# model parameters
			bm.write('bm = \'' + bench + '\';\n')
			bm.write('mdl_bef = \''+ model_bef + '\';\n')
			bm.write('mdl_rep = \''+ model_rep + '\';\n')
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

			if icc_name == []:
				bm.write('icc_name = {};\n')
			else:
				bm.write('icc_name = {\'' + icc_name[0] + '\'')
				for iccname in icc_name[1:]:
					bm.write(',')
					bm.write('\'' + iccname + '\'')
				bm.write('};\n')
				

			if ic_const == []:
				bm.write('ic_const = {};\n')
			else:
				bm.write('ic_const = {' + ic_const[0])
				for iccon in ic_const[1:]:
					bm.write(',')
					bm.write('' + iccon)
				bm.write('};\n')

			bm.write('ics_name = {\'' + ics_name[0] + '\'')
			for icsname in ics_name[1:]:
				bm.write(',')
				bm.write('\'' + icsname + '\'')
			bm.write('};\n')

			bm.write('ocb_name = {\'' + ocb_name[0] + '\'')
			for ocbname in ocb_name[1:]:
				bm.write(',')
				bm.write('\'' + ocbname + '\'')
			bm.write('};\n')

			bm.write('ocm_name = {\'' + ocm_name[0] + '\'')
			for ocmname in ocm_name[1:]:
				bm.write(',')
				bm.write('\'' + ocmname + '\'')
			bm.write('};\n')

			bm.write('ocf_name = {\'' + ocf_name[0] + '\'')
			for ocfname in ocf_name[1:]:
				bm.write(',')
				bm.write('\'' + ocfname + '\'')
			bm.write('};\n')

			bm.write('oc_range = {[' + str(oc_range[0][0]) + ' ' + str(oc_range[0][1]) + ']')
			for ocr in oc_range[1:]:
				bm.write(',[' + str(ocr[0]) + ' ' + str(ocr[1]) + ']')
			bm.write('};\n')

			bm.write('oc_span = ' + oc_span + ';\n')

			# switch setup 
			bm.write('global sw_sz sw patch_sig safe_patch_sig;\n')
			bm.write('sw_sz = T/Ts + 1;\n')
			bm.write('sw = ones(1, sw_sz);\n')
			bm.write('patch_sig = zeros(1, sw_sz);\n')
			bm.write('safe_patch_sig = zeros(1, sw_sz);\n')

			bm.write('phi_str = \'' + property[1] + '\';\n')
			bm.write('phi = STL_Formula(\'phi\',phi_str);\n')
			# phic_str can be a list of formulas separated by '||' in config.
			# IMPORTANT: keep the original spacing behavior (the original scripts used
			# strings with leading/trailing spaces between sub-formulas).
			if '||' in propertyc[1]:
				phic_items = propertyc[1].split('||')
				# emulate original formatting: add a leading space to items after the first,
				# and add a trailing space to items before the last.
				for idx in range(len(phic_items)):
					if idx > 0:
						phic_items[idx] = ' ' + phic_items[idx]
					if idx < len(phic_items) - 1:
						phic_items[idx] = phic_items[idx] + ' '
			else:
				phic_items = [propertyc[1]]
			# drop completely empty items (after removing surrounding whitespace only for emptiness check)
			phic_items = [s for s in phic_items if s.strip() != '']
			phic_items_escaped = [s.replace("'", "''") for s in phic_items]
			bm.write('phic_str = {' + ','.join(["'" + s + "'" for s in phic_items_escaped]) + '};\n')
			bm.write('interval_LR = {[' + str(interval_LR[0][0]) + ' ' + str(interval_LR[0][1]) + ']')
			for ilr in interval_LR[1:]:
				bm.write(',[' + str(ilr[0]) + ' ' + str(ilr[1]) + ']')
			bm.write('};\n')
		
			# hyper parameters for multi-obj optimization
			bm.write('pop_size = ' + pop_size + ';\n')	
			bm.write('max_gen = ' + max_gen + ';\n')
			bm.write('M = ' + M + ';\n')
			bm.write('V = ' + V + ';\n')
			bm.write('N = ' + N + ';\n')
			bm.write('use_custom_patch_points = ' + use_custom_patch_points + ';\n')
			bm.write('amp = ' + amp + ';\n')
			bm.write('st_id = ' + st_id + ';\n')
			bm.write('chroflag = ' + chroflag + ';\n')
			bm.write('chro_prefix = \'' + chro_prefix + '\';\n')
			bm.write('opt_pos = ' + opt_pos + ';\n')
			bm.write('t_l_ratio = ' + t_l_ratio + ';\n')
			bm.write('t_r_ratio = ' + t_r_ratio + ';\n')
			bm.write('use_custom_search_range = ' + use_custom_search_range + ';\n')
			bm.write('start_idx = ' + batch_start_str + ';\n')
			bm.write('end_idx = ' + batch_end_str + ';\n')
			bm.write('global failed_list;\n')
			bm.write('failed_list = [];\n')
			bm.write('rep_budget = ' + rep_budget + ';\n')
			# repair section
			bm.write('InitBreach;\n')
			bm.write('tic;\n')
			
			# Create a batch-specific model copy to avoid conflicts in parallel execution
			bm.write('mdl_rep_original = mdl_rep;\n')
			bm.write('mdl_rep_batch = [mdl_rep, \'_\', num2str(start_idx), \'_\', num2str(end_idx)];\n')
			bm.write('mdl_rep_path = which([mdl_rep_original, \'.slx\']);\n')
			bm.write('if isfile(mdl_rep_path)\n')
			bm.write('\tcopyfile(mdl_rep_path, [mdl_rep_batch, \'.slx\']);\n')
			bm.write('\tfprintf(\'Created batch-specific model copy: %s.slx\\n\', mdl_rep_batch);\n')
			bm.write('else\n')
			bm.write('\terror(\'Model file not found: %s.slx\', mdl_rep_original);\n')
			bm.write('end\n')
			
			bm.write(bench + ' = opera(bm, mdl_bef, mdl_rep_batch, D, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, ocb_name, ocm_name, ocf_name, oc_range, oc_span, phi, phic_str, interval_LR, pop_size, max_gen, M, V, N, use_custom_patch_points, amp, st_id, opt_pos, use_custom_search_range, t_l_ratio, t_r_ratio, chro_prefix, rep_budget);\n')
			bm.write('tr_in_cell = D.tr_in_cell;\n')
			bm.write('tr_ic_cell = D.tr_ic_cell;\n')
			bm.write('tr_oc_cell = D.tr_oc_cell;\n')
			bm.write('tr_rob_set = D.tr_rob_set;\n')
			bm.write('tr_tbr_num = 0;\n')
			bm.write('repair_log = {};\n')

			bm.write('for i = start_idx:end_idx\n')
			bm.write('\t if D.tr_rob_set(i,1) < 0\n')
			bm.write('\t\t tr_tbr_num = tr_tbr_num + 1;\n')
			bm.write('\t\t in_sig = D.tr_in_cell{1,i};\n')
			bm.write('\t\t [ic_sig_val, oc_sig_val, new_rob, rep_log] = ' + bench + '.sigRepair(in_sig, chroflag, i);\n')
			bm.write('\t\t if ~isempty(ic_sig_val) && ~isempty(oc_sig_val)\n')
			bm.write('\t\t\t for ti = 1: sw_sz\n')
			bm.write('\t\t\t\t sig_idx = numel(icc_name)+1;\n')
			bm.write('\t\t\t\t tr_ic_cell{1,ti}(sig_idx:end, i) = ic_sig_val(:, ti);\n')
			bm.write('\t\t\t\t tr_oc_cell{1,ti}(1,i) = oc_sig_val(1, ti);\n')
			bm.write('\t\t\t end\n')
			bm.write('\t\t end\n')
			bm.write('\t\t tr_rob_set(i,1) = new_rob;\n')
			bm.write('\t\t repair_log{1, end+1} = rep_log;\n')
			bm.write('\t\t close all;\n')
			bm.write('\t\t disp(i);\n')
			bm.write('\t\t disp(\'has been repaired\');\n')
			bm.write('\t else\n')
			bm.write('\t\t disp(i);\n')
			bm.write('\t\t disp(\'does not need repair\');\n')
			bm.write('\t end\n')
			bm.write('end\n')
			bm.write('toc;\n')
			bm.write('rep_time = toc;\n')
			
			# Generate output filename based on whether custom search range or custom patch points is used
			if use_custom_search_range == '1':
				rep_dataset_filename = '\'' + dataset_tbr[:-4] + '_St_'+ st_id + '_t_l_ratio_' + t_l_ratio + '_t_r_ratio_' + t_r_ratio + '_Start_' + batch_start_str + '_End_' + batch_end_str + '_Re.mat' + '\''
			elif use_custom_patch_points == '1':
				rep_dataset_filename = '\'' + dataset_tbr[:-4] + '_St_'+ st_id + '_N_' + N + '_Start_' + batch_start_str + '_End_' + batch_end_str + '_Re.mat' + '\''
			else:
				rep_dataset_filename = '\'' + dataset_tbr[:-4] + '_St_'+ st_id + '_Start_' + batch_start_str + '_End_' + batch_end_str + '_Re.mat' + '\''
			bm.write('rep_dataset_filename = ' + rep_dataset_filename + ';\n')
			bm.write('save(rep_dataset_filename, \'tr_in_cell\', \'tr_ic_cell\', \'tr_oc_cell\', \'tr_rob_set\', \'rep_time\', \'tr_tbr_num\', \'repair_log\', \'failed_list\');\n')
			bm.write('logname = ' + logname + ';\n')
			bm.write('sendEmail(logname);\n')
			bm.write('quit force\n')
			bm.write('EOF\n')