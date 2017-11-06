#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
#from diachronica1 import genphy
from genphy import genphy
from collections import defaultdict
import codecs
import numpy as np
from numpy import log,exp
from numpy.random import normal,uniform

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



def makemat(a,b,t): #generate transitional matrix from infinitesimal rates (faster than scipy.linalg, I think
    m = [[0,0],[0,0]]
    m[0][0] = (b/(a+b))+((a/(a+b))*exp(-1*(a+b)*t))
    m[0][1] = (a/(a+b))-((a/(a+b))*exp(-1*(a+b)*t))
    m[1][0] = (b/(a+b))-((b/(a+b))*exp(-1*(a+b)*t))
    m[1][1] = (a/(a+b))+((b/(a+b))*exp(-1*(a+b)*t))
    return m


def prune(a,b,feat,i): #compute tree likelihood under current gain and loss rates using pruning algorithm
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
    phi = log(np.dot(bindata[feat][pruneorders[i][-1]],[b/(a+b),a/(a+b)]))
    return phi


def inference(feat,chains=3,iters=10000):
    burnin=int(iters/2) #discard 1st half of samples
    thin=int(iters/100) #store only a 100th of samples
    posterior[feat]=defaultdict()
    acc_prop[feat]={}
    for c in range(chains):
        posterior[feat][c]=defaultdict(list)
        a = uniform(0,5) #initialize gain rate
        while a == 0:
            a = uniform(0,5)
        b = uniform(0,5) #initialize loss rate
        while b == 0:
            b = uniform(0,5)
        a_step = .5           #initialize step sizes for rate proposals
        b_step = .5
        accepted = []
        for i in range(len(pruneorders)):
            accepted_t = []
            lp_curr = prune(a,b,feat,i)
            for t in range(iters):
                a_prime = normal(a,a_step)
                while a_prime <= 0 or a_prime > 10:
                    a_prime = normal(a,a_step)
                b_prime = normal(b,b_step)
                while b_prime <= 0 or b_prime > 10:
                    b_prime = normal(b,b_step)
#                lp_curr = prune(a,b,feat,i)
                lp_prime = prune(a_prime,b_prime,feat,i)
                acc = uniform(0,1)
                if min(1,exp(lp_prime-lp_curr)) > acc:
                    a,b,lp_curr=a_prime,b_prime,lp_prime
#                    a,b=a_prime,b_prime
                    accepted_t.append(1)
                else:
                    accepted_t.append(0)
#                print t,i,a,b,lp_curr
                if t in range(thin,burnin,thin):
                    if sum(accepted_t)/len(accepted_t) < .22:
                        a_step *= exp(-.5)
                        b_step *= exp(-.5)
                    if sum(accepted_t)/len(accepted_t) > .25:
                        a_step *= exp(.5)
                        b_step *= exp(.5) 
                if t in range(burnin,iters,thin):
                    posterior[feat][c]['a'].append(a)
                    posterior[feat][c]['b'].append(b)
            accepted += accepted_t
        acc_prop[feat][c]=sum(accepted)/len(accepted)


def gelmandiag(f):
    return [(np.var(posterior[f][0]['a']+posterior[f][1]['a']+posterior[f][2]['a'])/((np.var(posterior[f][0]['a'])+np.var(posterior[f][1]['a'])+np.var(posterior[f][2]['a']))/3))**.5,(np.var(posterior[f][0]['b']+posterior[f][1]['b']+posterior[f][2]['b'])/((np.var(posterior[f][0]['b'])+np.var(posterior[f][1]['b'])+np.var(posterior[f][2]['b']))/3))**.5]



for feat in sorted(bindata.keys()):
    acc_prop = {}
    posterior=defaultdict()
    inference(feat)
    g = open('feature_rates.txt','a')
    for c in posterior[feat].keys():
        for d in posterior[feat][c].keys():
            for e in posterior[feat][c][d]:
                print >>g,feat+'\t'+str(c)+'\t'+d+'\t'+str(e)
    g.close()
    g = open('feature_Rhat.txt','a')
    print >>g, feat+'\t'+str(gelmandiag(feat)[0])+'\t'+str(gelmandiag(feat)[1])
    g.close()
    g = open('acceptance_probabilities.txt','a')
    for c in range(3):
      print >>g, feat+'\t'+str(c)+'\t'+str(acc_prop[feat][c])
    g.close()

