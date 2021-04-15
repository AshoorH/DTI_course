import numpy as np
import itertools
from copy import deepcopy
from math import exp
import os
from collections import defaultdict

def get_edge_list(sName):
	edge_list = []
	for line in open(sName).readlines():
		edge_list.append(line.strip().split())
	return edge_list

def get_adj_matrix_from_relation(aRelation,dDs,dTs):
	#print 'aRelation',aRelation
	adj = np.zeros((len(dDs.keys()),len(dTs.keys())))
	for element in aRelation:
		i = dDs[element[0]]
		j = dTs[element[1]]
		adj[i][j] = 1
	return adj

def make_sim_matrix(edge_list,dMap):
	sim = np.zeros((len(dMap.keys()),len(dMap.keys())))
	for a,b,c in edge_list:
		if float(c) > 0.0:	
			i = dMap[a]
			j = dMap[b]
			sim[i][j] = float(c)
			sim[j][i] = float(c)

		#fill diagonal with zeros
	np.fill_diagonal(sim,1)
	return sim

def get_similarities(sim_file,dMap,data_dir='data'):

    sim = []

    for line in open(sim_file).readlines():
        edge_list = get_edge_list(os.path.join(data_dir,line.strip()))
        sim.append(make_sim_matrix(edge_list,dMap))
    return sim

def get_D_T_info(R_all_train_test) :
    #print 'R_all_train_test',R_all_train_test
    
    D =[]
    T = []
    aAllPossiblePairs = []

    #R files
    with open(R_all_train_test, 'r') as f:
        data = f.readlines()
        #print 'data',data
        for line in data:
            #print 'line',line
            (a, b) = line.split()
            #print 'a,b',a,b
            D.append(a)
            T.append(b)


    #give labels for each Drugs and Targets
    D = list(set(D))
    T = list(set(T))


    dDs = dict([(x, j) for j, x in enumerate(D)])
    dTs = dict([(x, j) for j, x in enumerate(T)])
    diDs = dict([(j, x) for j, x in enumerate(D)])
    diTs = dict([(j, x) for j, x in enumerate(T)]) 
    #aAllPossiblePairs = list(itertools.product(D,T))
    return D,T,dDs,dTs,diDs,diTs

def xrange(x):

    return iter(range(x))


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)

        step = index.size//num
        for i in xrange(num):
            if i < num-1:
                #print(i*step,(i+1)*step)
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in xrange(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data

def mat2vec(mat):
    return list(mat.reshape((mat.shape[0]*mat.shape[1])))


def impute_zeros(inMat,inSim,k=5):
    
    mat = deepcopy(inMat)
    sim = deepcopy(inSim)
    (row,col) = mat.shape
    np.fill_diagonal(mat,0)


    indexZero = np.where(~mat.any(axis=1))[0]
    numIndexZeros = len(indexZero)

    np.fill_diagonal(sim,0)
    if numIndexZeros > 0:
        sim[:,indexZero] = 0
    for i in indexZero:
        currSimForZeros = sim[i,:]
        indexRank = np.argsort(currSimForZeros)

        indexNeig = indexRank[-k:]
        simCurr = currSimForZeros[indexNeig]

        mat_known = mat[indexNeig, :]
        
        
        if sum(simCurr) >0:  
            mat[i,: ] = np.dot(simCurr ,mat_known) / sum(simCurr)
        
    
    return mat


def func(x):
    return exp(-1*x)


def Get_GIP_profile(adj,t):
    '''It assumes target drug matrix'''

    bw = 1
    if t == "d": #profile for drugs similarity
        ga = np.dot(np.transpose(adj),adj)
    elif t=="t":
        ga = np.dot(adj,np.transpose(adj))
    else:
        sys.exit("The type is not supported: %s"%t)

    ga = bw*ga/np.mean(np.diag(ga))
    di = np.diag(ga)
    x =  np.tile(di,(1,di.shape[0])).reshape(di.shape[0],di.shape[0])
    #z = np.tile(np.transpose(di),(di.shape[0],1)).reshape(di.shape[0],di.shape[0])



    d =x+np.transpose(x)-2*ga
    
    f = np.vectorize(func)
    return f(d)

