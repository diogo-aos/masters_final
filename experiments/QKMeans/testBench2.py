'''
This version of the test bench is aimed to use with the Davies-Bouldin timings
from QK-Means and the early stop implementation. 
'''

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

import oracle
import qubitLib
import DaviesBouldin

import QK_Means
import K_Means

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="where to read data from",type=str)
parser.add_argument("-c", "--clusters", help="number of clusters",type=int)
parser.add_argument("-o", "--oracles", help="number of oracles",type=int)
parser.add_argument("-s", "--stringLen", help="length of qubit strings",type=int)
parser.add_argument("-g", "--generations", help="number of generations",type=int)
parser.add_argument("-r", "--rounds", help="number of rounds",type=int)
parser.add_argument("-f", "--factor", help="iteration factor for K-Means",type=float)
parser.add_argument("-e", "--earlyStop", help="early stop for QK-Means",type=int)
parser.add_argument("-ts", "--tailString", help="add tail string to end of all saved files",type=str)

args = parser.parse_args()

#########################################################
# file names
tailString=args.tailString

filenames=dict()
filenames['data']=args.data
filenames['qk'] = args.data[:-4] + '_' + 'qk' + tailString
filenames['k'] = args.data[:-4] + '_' + 'k' + tailString
filenames['params'] = args.data[:-4] + '_' + 'params' + tailString

# Quantum parameters
numClusters=args.clusters
numOracles=args.oracles
qubitStringLen=args.stringLen
qGenerations=args.generations
numRounds=args.rounds
earlyStop=args.earlyStop

# Normal parameters
initsPercentage=args.factor # multiplier factor for numOracles * qGenerations for the number of inits

# load data from file
pkl_file = open(filenames['data'], 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

mixture=data['data']
dim=data['dim']

params={'cluters':numClusters,'oracles':numOracles,'qbit length':qubitStringLen,
		'genetarions':qGenerations,'rounds':numRounds,'factor':initsPercentage,'inits':numInits,
        'early stop':earlyStop}

del data
del args
del parser

print "* * * * * * * * * * * * * * * * * * * * * * * * * * * *\n"
print "\t\t",filenames['data']
print "\n* * * * * * * * * * * * * * * * * * * * * * * * * * * *\n"


#########################################################

## Quantum K-Means

#########################################################

print 'Initiating QK-Means...'

qk_results=list()
qk_rounds_genNum=list() #list with the number of generations of each round
for i in range(numRounds):

    start = datetime.now()

    qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg,qk_db_timings=QK_Means.qk_means(mixture,numOracles,numClusters,qubitStringLen,qGenerations,dim,timeDB=True,earlyStop=earlyStop)
    qk_results.append([qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg,qk_db_timings])

    round=(datetime.now() - start).total_seconds()
    print float(i+1)*100/numRounds,'%\t','round ', i,':',round,'s  -  estimated:',(float(numRounds-1)-i)*round,'s / ',(float(numRounds-1)-i)*round/60,'m'

    qk_rounds_genNum.append(fitnessEvolution.shape[0]) #number of generations of current round



#########################################################
print 'Preparing data structures...'

qk_rounds=dict()
qk_rounds['centroids']=list()
qk_rounds['assignment']=list()
qk_rounds['fitness']=list()
qk_rounds['times']=list()
qk_rounds['DB times']=list()

for i in range(numRounds):
    qk_rounds['centroids'].append(qk_results[i][0])
    qk_rounds['assignment'].append(qk_results[i][1])
    qk_rounds['fitness'].append(qk_results[i][2])
    qk_rounds['times'].append(qk_results[i][3])
    qk_rounds['DB times'].append(qk_results[i][4])

    
# assign data to clusters
qk_rounds['assignedData']=[None]*numRounds #assigned data for the best solution in each round

for r in range(numRounds):
    # only assign for best solution
    best=int(qk_rounds['fitness'][r][-1,-1])
    qk_assignment=qk_rounds['assignment'][r]

    # store only best centroids
    qk_rounds['centroids'][r]=qk_rounds['centroids'][r][best]

    qk_assignedData = [None]*numClusters
    for i,j in enumerate(qk_assignment[best]):
        if qk_assignedData[j] != None:
            qk_assignedData[j] = np.vstack((qk_assignedData[j],mixture[i]))
        else:
            qk_assignedData[j] = mixture[i]
    
    qk_rounds['assignedData'][r]=qk_assignedData


# convert computation times
qk_rounds['total time'] = list()
qk_rounds['DB total time'] = list()
for i in range(numRounds):
    qk_times=qk_rounds['times'][i]
    qk_total = np.sum(np.array(qk_times))
    qk_rounds['total time'].append(qk_total)

    qk_db_times=qk_rounds['DB times'][i]
    qk_db_total=np.sum(np.array(qk_db_times))
    qk_rounds['DB total time'].append(qk_db_total)

# compute population's fitness evolution (best solution, mean, variance) per generation
qk_rounds['best evolution'] = [None]*numRounds
qk_rounds['pop variance'] = [None]*numRounds
qk_rounds['pop mean'] = [None]*numRounds

for i in range(numRounds):
    iGenerations=qk_rounds_genNum[i] #number of generations in round i

    bestSeq=[]
    fitnessEvolution=qk_rounds['fitness'][i] # fitness evolution of round i

    for j in range(0,iGenerations):
        bestIndex=fitnessEvolution[j,-1]
        bestSeq.append(fitnessEvolution[j,bestIndex])

    bestSeq=np.array(bestSeq)
    
    genVar=np.zeros(iGenerations)
    genMean=np.zeros(iGenerations)
    
    for j,ar in enumerate(fitnessEvolution):
        genVar[j]=np.var(ar[:-1])
        genMean[j]=np.mean(ar[:-1])
        
    qk_rounds['best evolution'][i]=bestSeq
    qk_rounds['pop variance'][i]=genVar
    qk_rounds['pop mean'][i]=genMean


qk_rounds['fitness best']=[None]*numRounds
qk_rounds['fitness worst']=[None]*numRounds
qk_rounds['fitness mean']=[None]*numRounds
qk_rounds['fitness variance']=[None]*numRounds

for i in range(numRounds):
	qk_rounds['fitness best'][i]=qk_rounds['best evolution'][i][-1]
	qk_rounds['fitness worst'][i]=np.max(qk_rounds['fitness'][i][:-1])
	qk_rounds['fitness mean'][i]=qk_rounds['pop mean'][i][-1]
	qk_rounds['fitness variance'][i]=qk_rounds['pop variance'][i][-1]

del qk_results
del qk_rounds['fitness']
del qk_rounds['times']
del qk_rounds['assignment']


#print "keys stored:"
#print qk_rounds.keys()

# save to file
output = open(filenames['qk']+'.pkl', 'wb')
pickle.dump(qk_rounds, output)
output.close()
print 'Data saved to '+filenames['qk']+'.pkl'

# save to file
output = open(filenames['params']+'.pkl', 'wb')
pickle.dump(params, output)
output.close()
print 'Parameters saved to ' + filenames['params'] + '.pkl'

#########################################################

print 'Initiating K-Means...'

k_results=list()

for i in range(numRounds):
    start=datetime.now()
    iGenerations=qk_rounds_genNum[i]

    numInits=np.int(iGenerations*numOracles*initsPercentage)

    k_centroids,k_assignment,k_timings,k_intertia=K_Means.k_means(mixture,numClusters,numInits)
    k_results.append([k_centroids,k_assignment,k_timings,k_intertia])

    round=(datetime.now() - start).total_seconds()
    print float(i+1)*100/numRounds,'%\t','round ', i,':',round,'s  -  estimated:',(float(numRounds-1)-i)*round,'s / ',(float(numRounds-1)-i)*round/60,'m'

print 'Preparing K-Means data structures...'

k_rounds=dict()
k_rounds['centroids']=list()
k_rounds['assignment']=list()
k_rounds['times']=list()
k_rounds['inertia']=list()

for i in range(numRounds):
    k_rounds['centroids'].append(k_results[i][0])
    k_rounds['assignment'].append(k_results[i][1])
    k_rounds['times'].append(k_results[i][2])
    k_rounds['inertia'].append(k_results[i][3])

# compute computation times
k_rounds['total time'] = list()

for i in range(numRounds):
    k_times=k_rounds['times'][i]
    k_total = np.sum(np.array(k_times))
    k_rounds['total time'].append(k_total)

#compute fitnesses
k_rounds['fitness best']=[np.inf]*numRounds
k_rounds['fitness worst']=[0]*numRounds
k_rounds['fitness mean']=[0]*numRounds
k_rounds['fitness variance']=[0]*numRounds
k_rounds['best centroids']=[None]*numRounds
k_rounds['fitness times']=[None]*numRounds
k_rounds['fitness best index']=[None]*numRounds

k_rounds['inertia best']=[np.inf]*numRounds


print 'Computing fitness score...'
for i in range(numRounds):
    start=datetime.now()

    roundFit=[None]*numInits #fitness of every initialization in current round
    roundFitBestIndex=None #index of initialization with best fitness

    for j in range(numInits):
        k_score=DaviesBouldin.DaviesBouldin(mixture,k_rounds['centroids'][i][j],k_rounds['assignment'][i][j])
        roundFit[j]=k_score.eval()

        # store best score & index
        if roundFit[j]<k_rounds['fitness best'][i]:
        	k_rounds['fitness best'][i]=roundFit[j]
        	roundFitBestIndex=j
        # store worst score
        if  roundFit[j]>k_rounds['fitness worst'][i]:
        	k_rounds['fitness worst'][i]= roundFit[j]

    k_rounds['fitness times'][i]=(datetime.now() - start).total_seconds()

    k_rounds['fitness best index'][i]=roundFitBestIndex

    # store fitness' mean and variance of current round
    k_rounds['fitness mean'][i]=np.mean(roundFit)
    k_rounds['fitness variance'][i]=np.var(roundFit)

    # store centroids of fittest solution
    k_rounds['best centroids'][i]=k_rounds['centroids'][i][roundFitBestIndex]

    round=(datetime.now() - start).total_seconds()
    print float(i+1)*100/numRounds,'%\t','round ', i,':',round,'s  -  estimated:',(float(numRounds-1)-i)*round,'s / ',(float(numRounds-1)-i)*round/60,'m'


k_rounds['centroids']=k_rounds['best centroids']

# assign data to clusters
k_rounds['assignedData']=[None]*numRounds #assigned data for the best solution in each round

for r in range(numRounds):
    # assign clusters to data
    k_assignment=k_rounds['assignment'][r][k_rounds['fitness best index'][r]]
    
    k_assignedData = [None]*numClusters
    for i,j in enumerate(k_assignment):
        if k_assignedData[j] != None:
            k_assignedData[j] = np.vstack((k_assignedData[j],mixture[i]))
        else:
            k_assignedData[j] = mixture[i]
    
    k_rounds['assignedData'][r]=k_assignedData

del k_results
del k_rounds['assignment']
del k_rounds['best centroids']
del k_rounds['times']
del k_rounds['fitness best index']

#print "keys stored:"
#print k_rounds.keys()

# save to file
output = open(filenames['k']+'.pkl', 'wb')
pickle.dump(k_rounds, output)
output.close()
print 'Data saved to '+filenames['k']+'.pkl'