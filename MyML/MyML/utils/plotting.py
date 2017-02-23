import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.figure

def plotThis(data,labels=None):

	if type(data) != type(list()):
		raise Exception("Data not a list.")

	if type(labels) != type(list()) and labels != None:
		raise Exception("Labels type not correct.")

	if len(data) == 0:
		raise Exception("No data")

	if labels != None and len(data)-1 != len(labels):
		raise Exception("Number of plots is different from number of labels.")
	
	x=data[0]
	data=data[1:]

	if labels == None:
		for d in data:
			plt.plot(x,d)
	else:
		for i,d in enumerate(data):
			plt.plot(x,d,label=labels[i])

		plt.legend(loc='best', framealpha=0.5, prop={'size':'small'})


def save_fig(fig, filename, format='eps', dpi=None):
	if not isinstance(fig,matplotlib.figure.Figure):
		raise Exception("fig not a figure")

	fig.savefig(filename + '.' + format, dpi=dpi, bbox_inches='tight')
