import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

import oracle
import qubit
from MyML.validation import DaviesBouldin

'''
Receives:
 - mixture              :   input data
 - numOracles           :   number of oracles to use
 - numClusters          :   number of clusters for K-Means
 - qubitStringLen       :   length of qubit strings
 - qGenerations         :   number of generations
 - dim                  :   dimension of data
 - timeDB               :   times Davies-Bouldin index computation;
                            default=False; if True, returns list DB timings
 - earlyStop            :   stops early when Davies-Bouldin fitness index is the
                            same a certain number of generations;
                            default=0 (no early stop)
Returns:
 - qk_centroids         :   centroids of oracles from last generation)
 - qk_assignment        :   assignment of oracles from last generation)
 - fitnessEvolution     :   matrix of DB index of every oracle over every
                            iteration
 - qk_timings_cg        :   timings of generations
 - db_timings           :   timings of Davies-Bouldin index computations
'''
def qk_means(mixture, numOracles, numClusters, qubitStringLen, qGenerations,
             dim, timeDB=False, earlyStop=0):
    # matrix for fitness evolution;
    # +1 for the column that indicates the best score in each gen
    fitnessEvolution = np.zeros((qGenerations,numOracles+1))

    if earlyStop != 0:
        useEarlyStop=True
        earlyStopCounter=0
    else:
        useEarlyStop=False


    if timeDB:
        db_timings = list() #timing list for Davies-Bouldin index computation

    qk_timings_cg = list()
    start = datetime.now()
    
    best = 0 #index of best oracle (starts at 0)
    
    oras = list()
    qk_centroids = [0]*numOracles
    qk_estimator = [0]*numOracles
    qk_assignment = [0]*numOracles
    
    for i in range(0,numOracles):
        oras.append(oracle.Oracle())
        oras[i].initialization(numClusters*dim,qubitStringLen)
        oras[i].collapse()
    
    qk_timings_cg.append((datetime.now() - start).total_seconds())
    start = datetime.now()
    
    for qGen_ in range(0,qGenerations):
        ## Clustering step
        for i,ora in enumerate(oras):
            if qGen_ != 0 and i == best: # current best shouldn't be modified
                continue 
            qk_centroids[i] = np.vstack(np.hsplit(ora.getIntArrays(),numClusters))
            qk_estimator[i] = KMeans(n_clusters=numClusters,init=qk_centroids[i],n_init=1)
            qk_assignment[i] = qk_estimator[i].fit_predict(mixture)
            qk_centroids[i] = qk_estimator[i].cluster_centers_
            ora.setIntArrays(np.concatenate(qk_centroids[i]))
        
        ## Compute fitness
            # start DB timing
            if timeDB:
                db_start = datetime.now()

            # compute DB
            score = DaviesBouldin.DaviesBouldin(mixture,qk_centroids[i],qk_assignment[i])
            ora.score = score.eval()

            # save timing
            if timeDB:
                db_timings.append((datetime.now() - db_start).total_seconds())

        
        ## Store best from this generation
        for i in range(1,numOracles):
            if oras[i].score < oras[best].score:
                best = i
                
        ## Quantum Rotation Gate 
        for i in range(0,numOracles):
            if i == best:
                continue
            
            oras[i].QuantumGateStep(oras[best])
            
        ## Collapse qubits
            oras[i].collapse()
            
        qk_timings_cg.append((datetime.now() - start).total_seconds())


        for i in range(0,numOracles):
            fitnessEvolution[qGen_,i]=oras[i].score
            fitnessEvolution[qGen_,-1]=best

        # check early stop
        if useEarlyStop:
            # increment counter if this generetion's fitness is the same as last
            if oras[best].score == fitnessEvolution[qGen_,fitnessEvolution[qGen_-1,-1]]:
                earlyStopCounter += 1
            else:
                earlyStopCounter == 0

            if earlyStopCounter == earlyStop:
                break

        start = datetime.now()

    # delete empty rows on fitness matrix
    emptyRows=range(qGen_+1,qGenerations)
    fitnessEvolution=np.delete(fitnessEvolution,emptyRows,axis=0)

    if timeDB:
        return qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg,db_timings

    return qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg