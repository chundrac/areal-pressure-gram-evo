#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
#from diachronica1 import genphy
from genphy import genphy
from collections import defaultdict
import codecs
import numpy as np
from numpy import log,exp
from numpy.random import normal,uniform,binomial
import random

langdict,revlangdict,locs,brlens,parents,children=genphy() #genrate node locations, branch lengths, parents, children, using ugly-looking function


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



tips = defaultdict(list)
for f in bindata.keys():
    for t in bindata[f].keys():
        tips[f].append(t)


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



def nodesim(n,a,b,feat,i): #fix this: missing data issue
    pi_n = [0,0]
    if n in children[i].keys() and n not in parents[i].keys(): #root
        pi_n[0] = (b/(a+b))*bindata[feat][n][0]
        pi_n[1] = (a/(a+b))*bindata[feat][n][0]
    if n in children[i].keys() and n in parents[i].keys(): #interior node
        pi_k = makemat(a,b,brlens[i][n])
        s_p = state[feat][parents[i][n]]
        pi_n[0] = pi_k[s_p][0]
        pi_n[1] = pi_k[s_p][1]
#        pi_n[0] *= bindata[feat][n][0]
#        pi_n[1] *= bindata[feat][n][1]
    if n in parents[i].keys() and n not in children[i].keys(): #tip
        pi_k = makemat(a,b,brlens[i][n])
        s_p = state[feat][parents[i][n]]
        pi_n[0] = pi_k[s_p][0]
        pi_n[1] = pi_k[s_p][1]
    Z = sum(pi_n)
#    pi_n[0]/=Z
    pi_n[1]/=Z
    pi[feat][n]=pi_n[1]
    s = binomial(1,pi[feat][n])
    state[feat][n] = s
    if n in tips[f]:
        states[feat][n].append(s)



def getcomanc(a,b):
    m,n = langdict[a],langdict[b]
    lineage_a = []
    lineage_b = []
    while m != '249':
        m = parents[0][m]
        lineage_a.append(m)
    while n != '249':
        n = parents[0][n]
        lineage_b.append(n)
    comanc = [x for x in lineage_a if x in lineage_b][0]
    return comanc

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
        updateorder = pruneorders[i][::-1]+tips[f]  #update nodes in order from root to leaves, including leaves for which there is missing data (pre-order traversal)
        for n in updateorder:    #update node state probabilities and draw states
            nodesim(n,a,b,f,i)
            
    

eval = defaultdict()
for f in bindata.keys():
    eval[f]=defaultdict(list)
    for t in tips[f]:
        for s in states[f][t]:
            if bindata[f][t] != (1,1):
                eval[f][t].append(1-abs(bindata[f][t].index(1)-s))


def poolavg():
    l = []
    for f in eval.keys():
        for k in eval[f].keys():
            l += eval[f][k]
    print 'mean accuracy of data regenerated at tips:',np.mean(l)


def avgavg():
    for f in sorted(eval.keys()):
        l = []
        for k in eval[f].keys():
            l += [np.mean(eval[f][k])]
        print ' & '.join(f.split('|')),'&','%.2f'%np.mean(l),'\\\\'


def langavg():
    langs = defaultdict(list)
    for f in sorted(eval.keys()):
        for k in eval[f].keys():
            langs[revlangdict[k]].append(np.mean(eval[f][k]))
    for k in sorted(langs.keys()):
        print k.replace('_',' ').replace(u'ç','\\c{c}').replace(u'č','\\v{c}').replace(u'å','\\r{a}'),'&','%.2f'%np.mean(langs[k]),'\\\\'


avgavg()
        
langavg()
