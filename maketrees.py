#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
#from diachronica1 import genphy
from genphy import genphy
from collections import defaultdict
import codecs
import numpy as np
from numpy import log,exp
from numpy.random import normal,uniform,binomial,exponential
from math import radians, cos, sin, asin, sqrt
import random
import re


langdict,revlangdict,locs,brlens,parents,children=genphy() #generate node locations, branch lengths, parents, children, using ugly-looking function


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
    for w in data[0][1:]:
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
        s_p = state[i][feat][parents[i][n]]
        pi_n[0] = bindata[feat][n][0]*pi_k[s_p][0]
        pi_n[1] = bindata[feat][n][1]*pi_k[s_p][1]
    if n in parents[i].keys() and n not in children[i].keys(): #tip
        pi_k = makemat(a,b,brlens[i][n])
        s_p = state[i][feat][parents[i][n]]
        pi_n[0] = bindata[feat][n][0]*pi_k[s_p][0]
        pi_n[1] = bindata[feat][n][1]*pi_k[s_p][1]
    Z = sum(pi_n)
#    pi_n[0]/=Z
    pi_n[1]/=Z
    pi[i][feat][n]=pi_n[1]
    s = binomial(1,pi[i][feat][n])
    state[i][feat][n] = s
    states[i][feat][n].append(s)



def gcdist(lat1,lon1,lat2,lon2): #function for computing greater-circle distance; BEAST gives lat lon coords, not lon lat
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r


#get node ages
def getage(n,i):
    a = []
    getparentage(n,a,i)
    return sum(a)


def getparentage(n,a,i):
    if n in parents[i].keys():
        p = parents[i][n]
        a.append(brlens[i][n]*1000)
        getparentage(p,a,i)


ages = []
for i in range(len(pruneorders)):
    age = {}
    age['249'] = 0
    for k in parents[i].keys():
        age[k] = getage(k,i)
    ages.append(age)


np.random.seed(0) #set seeds for replicability
random.seed(0)


#g = open('gainloss.csv','w')
    
#draw interior node states
iters=10
state = defaultdict()
states = defaultdict()
for i in range(18):
    states[i] = defaultdict()



for i in range(18):
    state[i] = defaultdict()


    
for t in range(iters):
    pi = {}
    feats = bindata.keys()
    for i in range(18):
      pi[i] = {}
      for f in feats:
        pi[i][f] = {}
        for k in bindata[f].keys():
            if bindata[f][k] == (0,1) or bindata[f][k] == (1,0):
                pi[i][f][k] = bindata[f][k][1]
        if f not in state[i].keys():
            state[i][f] = defaultdict(list)
        if f not in states[i].keys():
            states[i][f] = defaultdict(list)    
        a = random.sample(posterior[f][random.sample(xrange(3),1)[0]]['a'],1)[0]
        b = random.sample(posterior[f][random.sample(xrange(3),1)[0]]['b'],1)[0]
        prunenodes(a,b,f,i)
        observed = [k for k in bindata[f].keys() if bindata[f][k] == (0,1) or bindata[f][k] == (1,0)]
        for n in observed:
            states[i][f][n].append(bindata[f][n].index(1))
        updateorder = pruneorders[i][::-1]+[k for k in bindata[f].keys() if bindata[f][k] == (1,1)]  #update nodes in order from root to leaves, including leaves for which there is missing data (pre-order traversal)
        for n in updateorder:    #update node state probabilities and draw states
            nodestate(n,a,b,f,i)


def gentreegraph(i):
    graph = defaultdict(list)
    for n in pruneorders[i]:
        l = []
        for c in children[i][n]:
            if c not in graph.keys():
                l.append(int(c))
            else:
                l+=graph[c]
        graph[n]=[int(n),l]
    tree = graph['249']
    tree = str(tree)
    tree = re.split(r'([\[|\]|\,\ ])',tree)
    for i in range(len(tree))[::-1]:
        if tree[i] == '' or tree[i] == ' ' or tree[i] == ',':
            tree.pop(i)
    tree = tree[1:-1]
    treegraph = []
    j = 0
    for t in tree:
        if t=='[':
            j += 1
        if t==']':
            j -= 1
        if t != '[' and t != ']':
#    if t in revlangdict.keys():
#      treegraph.append(['-']*j+[revlangdict[t]])
#    else:
            treegraph.append(['-']*j+[t])
    return treegraph


#g = open('trees.txt','w')
for i in range(18):
  treegraph = gentreegraph(i)
  for f in sorted(states[i].keys()):
    print i,f
    for l in treegraph:
        for w in l:
            if w != '-':
                if w in revlangdict.keys():
                    print revlangdict[w],np.mean(states[i][f][w])
                else:
                    print np.mean(states[i][f][w])
            else:
                print w,
    print ''



#g.close()
