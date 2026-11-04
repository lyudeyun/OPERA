import os
import sys
# path
addpath = []
# dataset used for the model simulation
dataset = ''
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# D_name and D_size list the datasets to generate, in the same order (e.g. the retraining set
# ..._Tr.mat and the test set ..._Te.mat); one script generates all of them.
D_name = []
D_size = []
is_nor = ''
num_workers = '10'
# model parameters info
bench = ''
model = ''
parameters = []
# other configuration
in_name = []
in_range = []
in_span = []
icc_name = []
ic_const = []
ics_name = []
oc_name = []
oc_span = []
phi_str = []
# MC only: grid cells along position and velocity ("<n_pos> <n_vel>") and seed, one per dataset
grid = []
seed = []
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
			elif arg == 'D_name':
				D_name.append(argu[0])
			elif arg == 'D_size':
				D_size.append(argu[0])
			elif arg == 'is_nor':
				is_nor = argu[0]
			elif arg == 'num_workers':
				num_workers = argu[0]
			elif arg == 'bench':
				bench = argu[0]
			elif arg == 'model':
				model = argu[0]
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
			elif arg == 'icc_name':
				icc_name.append(argu[0])
			elif arg == 'ic_const':
				ic_const.append(argu[0])
			elif arg == 'ics_name':
				ics_name.append(argu[0])
			elif arg == 'oc_name':
				oc_name.append(argu[0])
			elif arg == 'oc_span':
				# print(len(argu))
				oc_span = '{'
				for idx in range(len(argu)-1):
					oc_span = oc_span + argu[idx]+','
					# print(argu[idx])
				oc_span = oc_span + argu[len(argu)-1] + '}'
			elif arg == 'grid':
				grid.append([int(argu[0]), int(argu[1])])
			elif arg == 'seed':
				seed.append(argu[0])
			elif arg == 'phi_str':
				complete_phi = argu[0]+';'+argu[1]
				for a in argu[2:]:
					complete_phi = complete_phi + ' ' + a
				phi_str.append(complete_phi)
			else:
				continue
			if linenum == 0:
				status = 0
def email_block(bm, logname):
	# Notify via Resend (Python); do not fail the job if email fails.
	bm.write('logname = ' + logname + ';\n')
	bm.write('try\n')
	bm.write('	[email_status, email_out] = system([\'python3 \', \'\'\'util/sendEmail.py\'\' \', \'\'\'\', char(logname), \'\'\'\']);\n')
	bm.write('	if email_status ~= 0\n')
	bm.write('		warning(\'opera:EmailSendFailed\', \'%s\', strtrim(email_out));\n')
	bm.write('	else\n')
	bm.write('		fprintf(\'%s\\n\', strtrim(email_out));\n')
	bm.write('	end\n')
	bm.write('catch ME\n')
	bm.write('	warning(\'opera:EmailSendFailed\', \'%s\', ME.message);\n')
	bm.write('end\n')

def script_name(names):
	# named after the first dataset, without its _Tr/_Te suffix when there are several
	name = os.path.splitext(os.path.basename(names[0]))[0]
	if len(names) > 1:
		for suffix in ('_Tr', '_Te'):
			if name.endswith(suffix):
				name = name[:-len(suffix)]
	return name

# MC is simulated in MATLAB without Simulink: each dataset is a seeded grid of initial states
# (benchmark/mc/genGridTestSet.m), with grid and seed given per dataset
if bench == 'MC':
	assert len(D_name) == len(D_size) == len(grid) == len(seed), 'D_name, D_size, grid and seed must list the same datasets'
	for (dn, ds, g) in zip(D_name, D_size, grid):
		assert int(ds) == g[0] * g[1], dn + ': D_size must equal the number of grid cells'
	dataset_name = script_name(D_name)
	filename = dataset_name + '_Gendata'
	print(filename)
	with open('scripts/' + filename, 'w') as bm:
		bm.write('#!/bin/sh\n')
		bm.write('csv=$1\n')
		bm.write('matlab -nodesktop -nosplash <<EOF\n')
		bm.write('clear;\n')
		bm.write('close all;\n')
		bm.write('clc;\n')
		bm.write('tic;\n')
		for ap in addpath:
			bm.write('addpath(genpath(\'' + ap + '\'));\n')
		bm.write('pool = gcp(\'nocreate\');\n')
		bm.write('if isempty(pool), parpool(' + num_workers + '); end\n')
		for (dn, g, sd) in zip(D_name, grid, seed):
			bm.write('genGridTestSet(' + str(g[0]) + ', ' + str(g[1]) + ', ' + sd + ', \'' + dn + '\', \'' + model + '\', ['
				+ str(in_range[0][0]) + ' ' + str(in_range[0][1]) + '], [' + str(in_range[1][0]) + ' ' + str(in_range[1][1]) + ']);\n')
		bm.write('gen_time = toc;\n')
		bm.write('save(\'' + dataset_name + '_Gendata_Time.mat\', \'gen_time\');\n')
		email_block(bm, '\'./output/' + filename + '.log\'')
		bm.write('quit force\n')
		bm.write('EOF\n')
	os.chmod('scripts/' + filename, 0o755)
	sys.exit(0)

# script name
for phi_i in range(len(phi_str)):
	property = phi_str[phi_i].split(';')
	assert len(D_name) == len(D_size), 'D_name and D_size must list the same number of datasets'
	# the script is named after the first dataset, without its _Tr/_Te suffix when there are several
	dataset_name = os.path.splitext(os.path.basename(D_name[0]))[0]
	if len(D_name) > 1:
		for suffix in ('_Tr', '_Te'):
			if dataset_name.endswith(suffix):
				dataset_name = dataset_name[:-len(suffix)]
	filename = dataset_name + '_Gendata'
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
		# load dataset
		if dataset != '':
			bm.write('D_run = load(\'' + dataset + '\');\n')
		# model parameters info
		bm.write('bm = \'' + bench + '\';\n')
		bm.write('is_nor = ' + is_nor + ';\n')
		bm.write('if is_nor == 1\n')
		bm.write('\tx_gain = D_run.ps_x.gain;\n')
		bm.write('\tx_gain = diag(x_gain);\n')
		bm.write('\tx_offset = D_run.ps_x.xoffset;\n')
		bm.write('\ty_gain = D_run.ps_y.gain;\n')
		bm.write('\ty_offset = D_run.ps_y.xoffset;\n')
		bm.write('end\n')

		bm.write('mdl = \''+ model + '\';\n')
		bm.write(param + '\n')

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

		bm.write('oc_name = {\'' + oc_name[0] + '\'')
		for ocname in oc_name[1:]:
			bm.write(',')
			bm.write('\'' + ocname + '\'')
		bm.write('};\n')
		bm.write('oc_span = ' + oc_span + ';\n')

		bm.write('phi_str = \'' + property[1] + '\';\n')
		bm.write('phi = STL_Formula(\'phi\',phi_str);\n')

		# dataset generation section
		bm.write('InitBreach;\n')
		for (dn, ds) in zip(D_name, D_size):
			bm.write('D_name = \'' + dn + '\';\n')
			bm.write('D_size = ' + ds + ';\n')
			if num_workers == '1':
				bm.write('generateDataSet(mdl, D_size, D_name, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, oc_name, phi);\n')
			else:
				bm.write('generateDataSetParallel(mdl, D_size, D_name, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, oc_name, phi, ' + num_workers + ');\n')
		bm.write('gen_time = toc;\n')
		time_filename = '\'' + dataset_name + '_Gendata_Time.mat' + '\''
		bm.write('time_filename = ' + time_filename + ';\n')
		bm.write('save(time_filename, \'gen_time\');\n')
		logname = '\'./output/' + filename + '.log\''
		bm.write('logname = ' + logname + ';\n')
		# Notify via Resend (Python); do not fail the job if email fails.
		bm.write('try\n')
		bm.write('	[email_status, email_out] = system([\'python3 \', \'\'\'util/sendEmail.py\'\' \', \'\'\'\', char(logname), \'\'\'\']);\n')
		bm.write('	if email_status ~= 0\n')
		bm.write('		warning(\'opera:EmailSendFailed\', \'%s\', strtrim(email_out));\n')
		bm.write('	else\n')
		bm.write('		fprintf(\'%s\\n\', strtrim(email_out));\n')
		bm.write('	end\n')
		bm.write('catch ME\n')
		bm.write('	warning(\'opera:EmailSendFailed\', \'%s\', ME.message);\n')
		bm.write('end\n')
		bm.write('quit force\n')
		bm.write('EOF\n')
	os.chmod('scripts/' + filename, 0o755)
