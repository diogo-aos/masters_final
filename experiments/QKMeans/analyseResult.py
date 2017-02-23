import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("-qk", "--qkResults", help="where to read Quantum K-Means results from",type=str)
parser.add_argument("-k", "--kResults", help="where to read K-Means results from",type=str)

args = parser.parse_args()

def readPickle(filename):
    # load data from file
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

data=[None]*2
data[0]=readPickle(args.qkResults)
data[1]=readPickle(args.kResults)
label=['QK-Means','K-Means']

## TIMING REPORT
print "\n*********************************************"
print "             TIMING REPORT"
print "*********************************************"
print 'Algorithm','\tmean','\t\tvariance','\t\tbest','\t\tworst'
for i,d in enumerate(data):
    print label[i],'\t',
    print np.mean(d['total time']),'\t',
    print np.var(d['total time']), '\t',
    print np.min(d['total time']), '\t',
    print np.max(d['total time'])
print label[i],'\t',
print np.mean(d['fitness times']),'\t',
print np.var(d['fitness times']),'\t',
print np.min(d['fitness times']),'\t',
print np.max(d['fitness times'])
print '(fit time)'



## FITNESS REPORT 
print "\n*********************************************"
print "             FITNESS REPORT"
print "*********************************************"
print "average of all rounds"
print 'Algorithm','\tbest','\t\tworst','\t\tmean','\t\tvariance', '\toverall best'
for i,d in enumerate(data):
    print label[i],'\t',
    print np.mean(d['fitness best']),'\t',
    print np.mean(d['fitness worst']),'\t',
    print np.mean(d['fitness mean']),'\t',
    print np.mean(d['fitness variance']),'\t',
    print np.min(d['fitness best'])


## QK-MEANS SPECIFIC
# FITNESS MEAN EVOLUTION
plt.figure()
for i,var in enumerate(data[0]['pop mean'][0:3]):
    plt.plot(range(100),var,label='round '+str(i))

plt.title('Population mean evolution on each round')
plt.xlabel('Generation')
plt.ylabel('Mean')


# FITNESS VARIANCE EVOLUTION
plt.figure()	
for i,var in enumerate(data[0]['pop variance'][0:3]):
    plt.plot(range(100),var,label='round '+str(i))
plt.title('Population variance evolution on each round')
plt.xlabel('Generation')
plt.ylabel('Variance')

# FITNESS BEST EVOLUTION
plt.figure()
for i,evo in enumerate(data[0]['best evolution']):
    plt.plot(range(100),evo,label='round '+str(i))
plt.title('Fitness evolution on each round')
plt.xlabel('Generation')
plt.ylabel('Fitness')

convGen=[0]*len(data[0]['best evolution'])
for i,evo in enumerate(data[0]['best evolution']):
    convGen[i]=np.argmin(evo)


print "\n*********************************************"
print "       QK-MEANS CONVERGENCE SPEED"
print "*********************************************"
print "convergence speed"
print 'mean','\tvariance','\tbest','\tworst'
print np.mean(convGen),'\t',
print np.var(convGen),'\t',
print np.min(convGen),'\t',
print np.max(convGen)

##


# if more than 2 dimensions, use PCA
use_pca=False
if data[0]['assignedData'][0][0].shape[1]>2:
	use_pca=True
	pca = PCA(n_components=2)

#pick the best round for each

for i,d in enumerate(data):
	plt.figure()

	#pick the best round
	bestRound=np.argmin(d['fitness best'])

	

	for j,g in enumerate(d['assignedData'][bestRound]):
		if use_pca:
			g=pca.fit_transform(g)
		plt.plot(g[:,0],g[:,1],'.')
plt.show()