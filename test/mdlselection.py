import sys
# path
addpath = []
# dataset
dataset = ''
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
# repair evaluation param 
trials = ''
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
			elif arg == 'trials':
				trials = argu[0]
			else:
				continue
			if linenum == 0:
				status = 0
# script name
for mdl in model:
	for phi_i in range(len(phi_str)):
		property = phi_str[phi_i].split(';')
		# executable script name
		filename = mdl + '_Selection'
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
			# load dataset
			if dataset != '':
				bm.write('D = load(\'' + dataset + '\');\n')
			# model parameters info
			bm.write('bm = \'' + bench + '\';\n')
			bm.write('mdl = \''+ mdl + '\';\n')
			bm.write(param + '\n')

			bm.write('is_nor = ' + is_nor + ';\n')
			bm.write('if is_nor == 1\n')
			bm.write('\tx_gain = D.ps_x.gain;\n')
			bm.write('\tx_gain = diag(x_gain);\n')
			bm.write('\tx_offset = D.ps_x.xoffset;\n')
			bm.write('\ty_gain = D.ps_y.gain;\n')
			bm.write('\ty_offset = D.ps_y.xoffset;\n')
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
			bm.write('trials = ' + trials + ';\n')
			bm.write('in_cell = {};\n')
			bm.write('rob_set = [];\n')
		
			# repair section
			bm.write('InitBreach;\n')
			bm.write('for i = 1:trials\n')
			bm.write('\tBr = BreachSimulinkSystem(mdl);\n')
			bm.write('\ttimespan = 0:Ts:T;\n')
			bm.write('\tBr.Sys.tspan = timespan;\n')
			bm.write('\tinput_gen.type = con_type;\n')
			bm.write('\tinput_gen.cp = T./cell2mat(in_span);\n')
			bm.write('\tBr.SetInputGen(input_gen);\n')
			bm.write('\tif strcmp(con_type, \'UniStep\')\n')
			bm.write('\t\ttemp_in_cell = {};\n')
			bm.write('\t\tfor in_i = 1:numel(in_name)\n')
			bm.write('\t\t\tin_range_i = in_range{1,in_i};\n')
			bm.write('\t\t\tin_sig = unifrnd(in_range_i(1,1), in_range_i(1,2), 1, input_gen.cp(1,1));\n')
			bm.write('\t\t\ttemp_in_cell{1,end+1} = in_sig;\n')
			bm.write('\t\t\tfor cpi = 0:input_gen.cp(1,1)-1\n')
			bm.write('\t\t\t\teval([\'Br.SetParam({\'\'\', ' + 'in_name{1,in_i}, ' + '\'_u\', ' +  'num2str(cpi), ' + '\'\'\'}, in_sig(1,cpi+1));\']);\n')
			bm.write('\t\t\tend\n')
			bm.write('\t\tend\n')
			bm.write('\tend\n')
			bm.write('\tBr.Sim(0:Ts:T);\n')
			bm.write('\trob = Br.CheckSpec(phi);\n')
			bm.write('\tin_cell{1,end+1} = temp_in_cell;\n')
			bm.write('\trob_set = [rob_set; rob];\n')
			bm.write('end\n')

			selection_filename = '\'' + mdl + '_Selection_Result.mat' + '\''
			bm.write('selection_filename = ' + selection_filename + ';\n')
			bm.write('save(selection_filename, \'in_cell\', \'rob_set\');\n')
			bm.write('quit force\n')
			bm.write('EOF\n')