import sys
# path
addpath = []
# training parameters
bench = ''
dataset = ''
x_name = ''
y_name = ''
is_nor = ''
retrain = ''
st_id = ''
nn_type = []
nn_stru = []
algo = []
date = ''
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
			elif arg == 'x_name':
				x_name = argu[0]
			elif arg == 'y_name':
				y_name = argu[0]
			elif arg == 'bench':
				bench = argu[0]
			elif arg == 'is_nor':
				is_nor = argu[0]
			elif arg == 'retrain':
				retrain = argu[0]
			elif arg == 'st_id':
				st_id = argu[0]
			elif arg == 'nn_type':
				nn_type.append(argu[0])
			elif arg == 'nn_stru':
				nn_stru.append(argu[0])
			elif arg == 'algo':
				algo.append(argu[0])
			elif arg == 'date':
				date = argu[0]
			else:
				continue
			if linenum == 0:
				status = 0
# script name 
for type in nn_type:
	for stru in nn_stru:
		for alg in algo:
			stru_copy = stru
			stru_copy = stru_copy.lstrip('[')
			stru_copy = stru_copy.rstrip(']')
			stru_copy = stru_copy.replace(',','_')
			print(stru_copy)
			if retrain == '0':
				filename = bench + '_' + type + '_' + alg + '_' + stru_copy + '_' + date + '_train'
			else:
				filename = bench + '_' + type + '_' + alg + '_' + stru_copy + '_st_' + st_id + '_' + date + '_retrain'
			#print(filename)
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
					bm.write('D = load(\'' + dataset + '\');\n')
				# model parameters info
				bm.write('bm = \'' + bench + '\';\n')
				bm.write('is_nor = ' + is_nor + ';\n')
				bm.write('nn_type = \'' + type + '\';\n')
				bm.write('nn_stru = ' + stru + ';\n')
				bm.write('algo = \'' + alg + '\';\n')
				bm.write('date = \'' + date + '\';\n')
				bm.write('st_id = ' + st_id + ';\n')
				bm.write('trainNNCtrl(bm, ' + x_name + ', ' + y_name + ', ' + 'nn_type, nn_stru, algo, is_nor, st_id, date);\n');
				bm.write('train_time = toc;\n')
				train_filename = '\'' + dataset + '_Train_Time.mat' + '\''
				bm.write('train_filename = ' + train_filename + ';\n')
				bm.write('save(train_filename, \'train_time\');\n')
				bm.write('quit force\n')
				bm.write('EOF\n')