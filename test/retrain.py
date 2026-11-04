import os
import sys
# path
addpath = []
# training parameters
bench = ''
train_mode = ''
dataset = []
dataset_ratio = []
x_name = ''
y_name = ''
is_nor = ''
st_list = []
st_list_ratio = []
nn_type = []
nn_stru = []
algo = []
max_epoch = ''
epoch_num = ''
ratio = ''
date = ''
start_i = ''
end_i = ''
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
				dataset.append(argu[0])
			elif arg == 'dataset_ratio':
				dataset_ratio.append(argu[0])
			elif arg == 'x_name':
				x_name = argu[0]
			elif arg == 'y_name':
				y_name = argu[0]
			elif arg == 'bench':
				bench = argu[0]
			elif arg == 'is_nor':
				is_nor = argu[0]
			elif arg == 'train_mode':
				train_mode = argu[0]
			elif arg == 'st_list':
				st_list.append(argu[0])
			elif arg == 'st_list_ratio':
				st_list_ratio.append(argu[0])
			elif arg == 'nn_type':
				nn_type.append(argu[0])
			elif arg == 'nn_stru':
				nn_stru.append(argu[0])
			elif arg == 'algo':
				algo.append(argu[0])
			elif arg == 'max_epoch':
				max_epoch = argu[0]
			elif arg == 'epoch_num':
				epoch_num = argu[0]
			elif arg == 'date':
				date = argu[0]
			elif arg == 'start_i':
				start_i = argu[0]
			elif arg == 'end_i':
				end_i = argu[0]
			else:
				continue
			if linenum == 0:
				status = 0

if train_mode == 'ratio':
	st_list = st_list_ratio
	dataset = dataset_ratio

# script name
for sti in range(len(st_list)):
	for type in nn_type:
		for stru in nn_stru:
			for alg in algo:
				for epi in range(1, int(epoch_num)+1):
					for ni in range(int(start_i), int(end_i)+1):
						stru_copy = stru
						stru_copy = stru_copy.lstrip('[')
						stru_copy = stru_copy.rstrip(']')
						stru_copy = stru_copy.replace(',','_')
						epoch = int(epi * int(max_epoch)/int(epoch_num))
						# decide the filename according to the ratio
						if 'ratio' in dataset[sti]:
							substring = dataset[sti].split("_")[-1]
							ratio = substring[:-4]
							filename = bench + '_' + type + '_' + alg + '_' + stru_copy + '_St_' + st_list[sti] + '_epoch_' + str(int(epoch)) + '_ratio_' + ratio + '_' + date + '_retrain_' + str(ni) 
						else:
							ratio = '-1'
							filename = bench + '_' + type + '_' + alg + '_' + stru_copy + '_St_' + st_list[sti] + '_epoch_' + str(int(epoch)) + '_' + date + '_retrain_' + str(ni)
						print(filename)
						with open('scripts/'+filename,'w') as bm:
							bm.write('#!/bin/sh\n')
							bm.write('csv=$1\n')
							bm.write('matlab -nodesktop -nosplash <<EOF\n')
							bm.write('clear;\n')
							bm.write('close all;\n')
							bm.write('clc;\n')
							bm.write('bdclose(\'all\');\n')
							bm.write('maxNumCompThreads(12);\n')
							# addpath
							for ap in addpath:
								bm.write('addpath(genpath(\'' + ap + '\'));\n')
							# load dataset
							# D = dataset[sti-1]
							if dataset[sti] != '':
								bm.write('D = load(\'' + dataset[sti] + '\');\n')
							# model parameters
							bm.write('bench = \'' + bench + '\';\n')
							bm.write('is_nor = ' + is_nor + ';\n')
							bm.write('nn_type = \'' + type + '\';\n')
							bm.write('nn_stru = ' + stru + ';\n')
							bm.write('algo = \'' + alg + '\';\n')
							bm.write('date = \'' + date + '\';\n')
							bm.write('sti = ' + st_list[sti] + ';\n')
							bm.write('epoch = ' + str(epoch) + ';\n')
							# we can obtain ratio from the dataset name
							bm.write('ratio = ' + str(ratio) + ';\n')
							bm.write('trainNNCtrl(bench, ' + x_name + ', ' + y_name + ', ' + 'nn_type, nn_stru, algo, is_nor, epoch, ratio, sti, date, ' + str(ni) + ');\n');
							bm.write('quit force\n')
							bm.write('EOF\n')
						os.chmod('scripts/' + filename, 0o755)