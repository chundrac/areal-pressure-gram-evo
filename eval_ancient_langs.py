#!usr/bin/env python
# -*- coding: utf-8 -*-

from  __future__ import division
#from diachronica1 import genphy
from genphy import genphy
from collections import defaultdict
import codecs
import numpy as np
from numpy import log,exp
from numpy.random import normal,uniform,binomial
import random

langdict,revlangdict,locs,brlens,parents,children=genphy() #genrate node locations, branch lengths, parents, children, using ugly-looking function


#revlangdict = {}
#for k in langdict.keys():
#    revlangdict[langdict[k]]=k


testlist = []
for k in brlens[0]:
    if brlens[0][k] < .01 and k in revlangdict.keys():
        testlist.append(k)



def genpruneorder(): #arrange branches of tree i in proper order for post-order traversal of tree (cf. Felsenstein 1981)
    pruneorders = []
    for i in range(len(children)): #for all trees in sample do
        pruneorder = langdict.values() #start with tips of tree
        nodes = children[i].keys() #collect every node that is a parent, to be appended to object pruneorder
        while len(nodes) > 0: #as long as there are still items in "nodes"
            for k in nodes:
                if k not in pruneorder and children[i][k][0] in pruneorder and children[i][k][1] in pruneorder:
                    pruneorder.append(k)
                    nodes.pop(nodes.index(k)) #put node into object pruneorder if it is not already there, and both its children are
        pruneorder = pruneorder[len(langdict.keys()):] #remove tips from pruneorder
        pruneorders.append(pruneorder)
#    prunebranches = []
#    for n in pruneorder:
#        for l in branchmatrix[i]:
#            if l[0] == n:
#                prunebranches.append(l)
    return pruneorders


pruneorders = genpruneorder()
def binarizedata(): #convert binary tip data into tuples (0,1), (1,0), (1,1) (if missing, cf. Felsenstein 2004, ch. 16)
    data = []
    for l in open('lundic_matrix_1130.csv','r'):
#    for l in open('lundic_matrix.csv','r'):
        data.append(l.strip().split('\t'))
    for i in range(len(data)):
        data[i][0]=codecs.decode(data[i][0],'utf8').replace('(','').replace(')','')
#    print data[0]
    bindata={}
    for w in data[0][3:]:
        bindata[w]={}
    for l in data[1:]:
        if l[0] in langdict.keys():
            for i in range(3,len(l)):
                if l[i] == '1':
                    bindata[data[0][i]][langdict[l[0]]] = (0,1)
                if l[i] == '0':
                    bindata[data[0][i]][langdict[l[0]]] = (1,0)
                if l[i] == 'NA':
                    bindata[data[0][i]][langdict[l[0]]] = (1,1)
    return bindata


bindata = binarizedata()
for f in bindata.keys(): #get rid of features that do not differ across IE languages
    if (0,1) not in bindata[f].values() or (1,0) not in bindata[f].values():
#        print f
        bindata.pop(f)


testset = {}
for f in bindata.keys():
    testset[f] = {}
    for l in testlist:
        testset[f][l] = bindata[f][l].index(1)
        bindata[f][l] = (1,1)


posterior = defaultdict()
for line in open('feature_rates.txt','r'):
    l = line.split('\t')
    if l[0] not in posterior.keys():
        posterior[l[0]]=defaultdict()
        posterior[l[0]][0]=defaultdict(list)
        posterior[l[0]][1]=defaultdict(list)
        posterior[l[0]][2]=defaultdict(list)
    posterior[l[0]][int(l[1])][l[2]].append(float(l[3]))



def makemat(a,b,t): #generate transitional matrix from infinitesimal rates (faster than scipy.linalg, I think
    m = [[0,0],[0,0]]
    m[0][0] = (b/(a+b))+((a/(a+b))*exp(-1*(a+b)*t))
    m[0][1] = (a/(a+b))-((a/(a+b))*exp(-1*(a+b)*t))
    m[1][0] = (b/(a+b))-((b/(a+b))*exp(-1*(a+b)*t))
    m[1][1] = (a/(a+b))+((b/(a+b))*exp(-1*(a+b)*t))
    return m


def prunenodes(a,b,feat,i): #compute tree likelihood under current gain and loss rates using pruning algorithm
    for n in pruneorders[i]:
        pi = [0,0]
        for k in children[i][n]:
            pi_k = makemat(a,b,brlens[i][k])
            pi_k[0][0] *= bindata[feat][k][0]
            pi_k[1][0] *= bindata[feat][k][0]
            pi_k[0][1] *= bindata[feat][k][1]
            pi_k[1][1] *= bindata[feat][k][1]
            pi[0] += log(sum(pi_k[0]))
            pi[1] += log(sum(pi_k[1]))
        bindata[feat][n] = tuple(exp(pi))
#    phi = log(np.dot(bindata[feat][pruneorders[i][-1]],[b/(a+b),a/(a+b)]))
#    return phi



def nodestate(n,a,b,feat,i): #fix this: missing data issue
    pi_n = [0,0]
    if n in children[i].keys() and n not in parents[i].keys(): #root
        pi_n[0] = (b/(a+b))*bindata[feat][n][0]
        pi_n[1] = (a/(a+b))*bindata[feat][n][0]
    if n in children[i].keys() and n in parents[i].keys(): #interior node
        pi_k = makemat(a,b,brlens[i][n])
        s_p = state[feat][parents[i][n]]
        pi_n[0] = bindata[feat][n][0]*pi_k[s_p][0]
        pi_n[1] = bindata[feat][n][1]*pi_k[s_p][1]
    if n in parents[i].keys() and n not in children[i].keys(): #tip
        pi_k = makemat(a,b,brlens[i][n])
        s_p = state[feat][parents[i][n]]
        pi_n[0] = bindata[feat][n][0]*pi_k[s_p][0]
        pi_n[1] = bindata[feat][n][1]*pi_k[s_p][1]
    Z = sum(pi_n)
#    pi_n[0]/=Z
    pi_n[1]/=Z
    pi[feat][n]=pi_n[1]
    s = binomial(1,pi[feat][n])
    state[feat][n] = s
    states[feat][n].append(s)


random.seed(0)
np.random.seed(0)


iters=100
states = defaultdict()
state = {}
for t in range(iters):
    pi = {}
    feats = bindata.keys()
    for f in feats:
        pi[f] = {}
        for k in bindata[f].keys():
            if bindata[f][k] == (0,1) or bindata[f][k] == (1,0):
                pi[f][k] = bindata[f][k][1]
        if f not in state.keys():
            states[f] = defaultdict(list)
            state[f] = {}
        a = random.sample(posterior[f][random.sample(xrange(3),1)[0]]['a'],1)[0]
        b = random.sample(posterior[f][random.sample(xrange(3),1)[0]]['b'],1)[0]
        i = random.sample(xrange(len(pruneorders)),1)[0]
        prunenodes(a,b,f,i)
        updateorder = pruneorders[i][::-1]+[k for k in bindata[f].keys() if bindata[f][k] == (1,1)]  #update nodes in order from root to leaves, including leaves for which there is missing data (pre-order traversal)
        for n in updateorder:    #update node state probabilities and draw states
            nodestate(n,a,b,f,i)
            

scores = []
for f in bindata.keys():
    for l in testlist:
#        print f,'&',revlangdict[l],1-abs(np.mean(states[f][l])-testset[f][l]),'\\\\'
        scores.append(1-abs(np.mean(states[f][l])-testset[f][l]))

print 'Mean Accuracy','&','%.2f'%np.mean(scores),'\\\\'
print '\\hline'
for l in sorted(testlist,key=lambda(x):revlangdict[x]):
    score = []
    for f in bindata.keys():
        score.append(1-abs(np.mean(states[f][l])-testset[f][l]))
    print revlangdict[l].replace('_',' ').replace(u'ç','\\c{c}'),'&','%.2f'%np.mean(score),'\\\\'
