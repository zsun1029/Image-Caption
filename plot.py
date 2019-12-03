# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
import numpy as np

def draw_val_all():
	x_val = range(len(fval))
	y_val_IDEr = [float(line[5]) for line in fval] 
	y_val_Blue4 = [float(line[7]) for line in fval] 
	y_val_ROUGEL = [float(line[15]) for line in fval]
	y_val_METEOR = [float(line[17]) for line in fval]
	plt.plot(x_val, y_val_METEOR, marker='.',   label=u'METEOR')
	plt.plot(x_val, y_val_Blue4, marker='o',  label=u'Blue4')
	plt.plot(x_val, y_val_ROUGEL, marker='+', label=u'ROUGEL')
	plt.plot(x_val, y_val_IDEr, marker='*',  label=u'IDEr')
	plt.legend()  # 让图例生效
	xnames = [i for i in x_val] 
	plt.xticks(x_val, xnames, rotation=45) 
	plt.margins(0)
	plt.subplots_adjust(bottom=0.1)
	plt.xlabel(u"epoch") #X轴标签
	plt.ylabel("value") #Y轴标签
	plt.title("各个评测值的变化曲线") #标题 
	plt.show()

def draw_valtest():
	x_val = range(len(fval))
	y_val_IDEr = [float(line[5]) for line in fval]
	y_test_IDEr = [float(line[5]) for line in ftest] 
	plt.plot(x_val, y_val_IDEr, marker='*', label=u'val')
	plt.plot(x_val, y_test_IDEr, marker='o', label=u'test')
	plt.legend()  # 让图例生效
	y_val = range(1) 
	ynames = np.arange(0.6, 1.0, 0.025)
	plt.xticks(x_val, x_val, rotation=45)
	# plt.yticks(ynames, ynames, rotation=45)
	plt.margins(0)
	plt.subplots_adjust(bottom=0.15)
	plt.xlabel(u"epoch") #X轴标签  , size=14
	plt.ylabel("value") #Y轴标签
	plt.title("val/test在IDEr评测值下的对比曲线图") #标题 
	plt.show()

def draw_allmodel():
	f0 = open("info_onlydecoder_batch_16.log", 'r').readlines()
	f0 = [line.split() for line in f0 if len(line)>100]
	fval0 = [line for line in f0 if line[0] == 'val'] 
	f1 = open("info_1_batch_32.log", 'r').readlines()
	f1 = [line.split() for line in f1 if len(line)>100]
	fval1 = [line for line in f1 if line[0] == 'val'] 
	f2 = open("info_2_batch_16.log", 'r').readlines()
	f2 = [line.split() for line in f2 if len(line)>100]
	fval2 = [line for line in f2 if line[0] == 'val']  
	f41 = open("info_14_batch_16.log", 'r').readlines()
	f41 = [line.split() for line in f41 if len(line)>100]
	fval41 = [line for line in f41 if line[0] == 'val']  
	f42 = open("info_1_batch_16.log", 'r').readlines()
	f42 = [line.split() for line in f42 if len(line)>100]
	fval42 = [line for line in f42 if line[0] == 'val']  

	x_val = range(80)
	y_val0 = [float(line[5]) for line in fval0] 
	y_val1 = [float(line[5]) for line in fval1] 
	y_val2 = [float(line[5]) for line in fval2] 
	y_val3 = [float(line[5]) for line in fval3]
	y_val41 = [float(line[5]) for line in fval41]
	y_val42 = [float(line[5]) for line in fval42]
	plt.plot(range(len(y_val0)), y_val0, linestyle='--',   label=u'info_0')
	plt.plot(range(len(y_val1)), y_val1, marker='.',   label=u'info_1')
	plt.plot(range(len(y_val2)), y_val2, marker='o',  label=u'info_2')
	plt.plot(range(len(y_val3)), y_val3, marker='+', label=u'info_3')
	plt.plot(range(len(y_val41)), y_val41, marker='*',  label=u'info_41') 
	plt.plot(range(len(y_val42)), y_val42, marker='.',  label=u'info_42') 
	plt.legend()  # 让图例生效 
	plt.xticks(x_val, x_val, rotation=45) 
	plt.margins(0)
	plt.subplots_adjust(bottom=0.1)
	plt.xlabel(u"epoch") #X轴标签
	plt.ylabel("value") #Y轴标签
	plt.title("各个评测值的变化曲线") #标题 
	plt.show()



f3 = open("info_3_batch_16.log", 'r').readlines()
f3 = [line.split() for line in f3 if len(line)>100]
fval3 = [line for line in f3 if line[0] == 'val']
ftest3 = [line for line in f3 if line[0] == 'test']
# draw_val_all()
# draw_valtest()

draw_allmodel()
 
# import matplotlib
# matplotlib.use('Agg')  

# xData = np.arange(0, 10, 1)
# yData1 = xData.__pow__(2.0)
# yData2 = np.arange(15, 61, 5)
# plt.figure(num=1, figsize=(8, 6))
# plt.plot(xData, yData1, color='b', linestyle='--', marker='o', label='y1 data')
# plt.plot(xData, yData2, color='r', linestyle='-', label='y2 data')
# # plt.legend(loc='upper left') 
# plt.legend()
# plt.show()
# if not os.path.exists('images'):
#     os.mkdir('images')
# plt.savefig('images/plot1.png', format='png')