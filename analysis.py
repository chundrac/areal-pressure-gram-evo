#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
#from diachronica1 import genphy
from genphy import genphy
from collections import defaultdict
import codecs
import numpy as np
from numpy import log,exp
from numpy.random import normal,uniform,binomial,exponential
from math import radians, cos, sin, asin, sqrt
import random


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


condition = {}
for line in open('condition.csv','r'):
    l = line.strip().split('\t')
    condition[l[0]]=tuple(l[1:])


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


def nodestate(n,a,b,feat,i,pi,state,states): #fix this: missing data issue
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


lineage = defaultdict()
def getanc(i,n,l):
  if n in parents[i].keys():
    m = parents[i][n]
    l.append(m)
    getanc(i,m,l)

def getlin(i,n):
  lin = []
  getanc(i,n,lin)
  return lin


for i in range(len(pruneorders)):
  lineage[i] = defaultdict(list)
  for m in parents[i].keys()+['249']:
    lineage[i][m]=getlin(i,m)



cophens = []
for i in range(len(pruneorders)):
  cophen = {}                       #KEEP!!! GET COPHENETIC DISTANCES
  for k in lineage[i].keys():
    for j in lineage[i].keys():
      if k!=j:
        x = lineage[i][k]
        y = lineage[i][j]
        com = []
        cdist = 0
        if k not in y and j not in x:
          comx = [j]
          for a in x:
            if a not in y:
              comx.append(a)
          com.append(comx)
          comy = [k]
          for a in y:
            if a not in x:
              comy.append(a)
          com.append(comy)
          for c in com:
            for l in c:
              cdist+=brlens[i][l]*1000
          cdist=cdist
          cophen[(k,j)]=cdist
        if k in y:
          com.append(j)
          for i in range(0,y.index(k)):
            com.append(y[i])
          for l in com:
            cdist+=brlens[i][l]*1000
          cophen[(k,j)]=cdist
        if j in x:
          com.append(k)
          for i in range(0,x.index(j)):
            com.append(x[i])
          for l in com:
            cdist+=brlens[i][l]*1000
          cophen[(k,j)]=cdist
  cophens.append(cophen)



    

def getdistances(i,tuple1,tuple2):
    node1 = tuple1[0]
    node2 = tuple2[0]
    t1 = tuple1[1]
    t2 = tuple2[1]
    p1 = parents[i][node1]
    p2 = parents[i][node2]
    abst1 = ages[i][p1] + t1
    abst2 = ages[i][p2] + t2   #absolute times
    prop1 = t1/brlens[i][node1]
    prop2 = t2/brlens[i][node2]
    lat1 = locs[i][parents[i][node1]][0] + ((locs[i][parents[i][node1]][0] - locs[i][node1][0])*prop1)
    lon1 = locs[i][parents[i][node1]][1] + ((locs[i][parents[i][node1]][1] - locs[i][node1][1])*prop1)
    lat2 = locs[i][parents[i][node2]][0] + ((locs[i][parents[i][node2]][0] - locs[i][node2][0])*prop2)
    lon2 = locs[i][parents[i][node2]][1] + ((locs[i][parents[i][node2]][1] - locs[i][node2][1])*prop2)
    cophdist = cophens[i][(node1,node2)] - (t1 + t2)/2
    geodist = ((gcdist(lat1,lon1,lat2,lon2)**2+(abst2-abst1)**2)**.5) #spatiotemporal distance
#    geodist = gcdist(lat1,lon1,lat2,lon2)
    return geodist,cophdist
    


def gendistances(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    g = open('change_distances.csv','w')
    print >>g, "iter\ttree\tgrid\tfeat\tvar\tchange\tnchange\tgeodist\tcophdist\tcondition"
    iters = 100
    state = defaultdict()
    states = defaultdict()
    for t in range(iters):
            pi = {}
            feats = bindata.keys()
            for f in sorted(feats):
                pi[f] = {}
                for k in bindata[f].keys():
                    if bindata[f][k] == (0,1) or bindata[f][k] == (1,0):
                        pi[f][k] = bindata[f][k][1]
                if f not in state.keys():
                    state[f] = defaultdict(list)
                if f not in states.keys():
                    states[f] = defaultdict(list)    
                a = random.sample(posterior[f][random.sample(xrange(3),1)[0]]['a'],1)[0]
                b = random.sample(posterior[f][random.sample(xrange(3),1)[0]]['b'],1)[0]
                i = random.sample(xrange(len(pruneorders)),1)[0]
                prunenodes(a,b,f,i)
                observed = [k for k in bindata[f].keys() if bindata[f][k] == (0,1) or bindata[f][k] == (1,0)]
                for n in observed:
                    states[f][n].append(bindata[f][n].index(1))
                updateorder = pruneorders[i][::-1]+[k for k in bindata[f].keys() if bindata[f][k] == (1,1)]  #update nodes in order from root to leaves, including leaves for which there is missing data (pre-order traversal)
                for n in updateorder:    #update node state probabilities and draw states
                    nodestate(n,a,b,f,i,pi,state,states)
                #put distance arrays here
                gains = []
                losses = []
                for k in parents[i].keys(): #for each branch
                    p = parents[i][k]
                    if states[f][k][t] == 1 and states[f][p][t] == 0: #gain on branch?
                        wait_time = exponential(a)                    #draw waiting time
                        while wait_time >= brlens[i][k]:
                            wait_time = exponential(a)
                        abs_time = ages[i][p]+wait_time
                        gains.append((k,wait_time))
                    if states[f][k][t] == 0 and states[f][p][t] == 1: #loss on branch?
                        wait_time = exponential(b)
                        while wait_time >= brlens[i][k]:
                            wait_time = exponential(b)
                        abs_time = ages[i][p]+wait_time
                        losses.append((k,wait_time))
                gain_geodists = [] #nearest geographic neighbor distances for each gain event
                gain_cophdists = [] #corresponding cophenetic distances between gain events
                if len(gains) > 1:
                    for l in gains:
                        gains_geodists_k = [] #distances between gain event k and all other gain events
                        gains_cophdists_k = [] #corresponding cophenetic distances
                        for j in gains:
                            if l != j:
                                geo,coph = getdistances(i,l,j)
                                gains_geodists_k.append(geo)
                                gains_cophdists_k.append(coph)
                        if len(gains_geodists_k) > 0:
                            min_geodist = min(gains_geodists_k)
                            cophdist = gains_cophdists_k[gains_geodists_k.index(min_geodist)]
                            gain_geodists.append(min_geodist)
                            gain_cophdists.append(cophdist)
                    print >>g, str(t)+'\t'+str(i)+'\t'+'\t'.join(f.split('|'))+'\t'+'gain'+'\t'+str(len(gains))+'\t'+str(np.mean(gain_geodists))+'\t'+str(np.mean(gain_cophdists))+'\t'+str(condition[f][0])
                loss_geodists = [] #nearest geographic neighbor distances for each loss event
                loss_cophdists = [] #corresponding cophenetic distances between loss events
                if len(losses) > 1:
                    for l in losses:
                        losses_geodists_k = [] #distances between gain event k and all other loss events
                        losses_cophdists_k = [] #corresponding cophenetic distances
                        for j in losses:
                            if l != j:
                                geo,coph = getdistances(i,l,j)
                                losses_geodists_k.append(geo)
                                losses_cophdists_k.append(coph)
                        if len(losses_geodists_k) > 0:
                            min_geodist = min(losses_geodists_k)
                            cophdist = losses_cophdists_k[losses_geodists_k.index(min_geodist)]
                            loss_geodists.append(min_geodist)
                            loss_cophdists.append(cophdist)
                    print >>g, str(t)+'\t'+str(i)+'\t'+'\t'.join(f.split('|'))+'\t'+'loss'+'\t'+str(len(losses))+'\t'+str(np.mean(loss_geodists))+'\t'+str(np.mean(loss_cophdists))+'\t'+str(condition[f][1])
    g.close()



def main():
    if len(sys.argv) < 2:
        random_seed = 0
        print 'usage: python simulate_distances.py [seed]\nsetting seed to default of 0'
    else:
        random_seed = int(sys.argv[1])
    gendistances(random_seed)


if __name__ == "__main__":
    main()
