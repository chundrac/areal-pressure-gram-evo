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

langdict,revlangdict,locs,brlens,parents,children=genphy() 


featdep="""Nominal morphology|Case marking|CASE-FIRST    1
Nominal morphology|Case marking|CASE-LAST    1

Nominal morphology|Definiteness marking|DEF.ART    0    0    0    0    1    1    1    1    0    0
Nominal morphology|Definiteness marking|N-DEF    0    0    1    1    0    0    1    1    0    0
Nominal morphology|Definiteness marking|ADJ-DEF    0    1    0    1    0    1    0    1    0    0
Nominal morphology|Definiteness marking|DEF-LAST    1    1    1    1    1    1    1    1    0    1
Nominal morphology|Definiteness marking|DEF-FIRST    1    1    1    1    1    1    1    1    1    0

Nominal morphology|Nominal cases|GEN/DAT    1    1    1    1
Nominal morphology|Nominal cases|DAT    1    0    1    0
Nominal morphology|Nominal cases|GEN    1    1    0    1

Verbal morphology|present progressive, A|PROG:NO-A-AGR    1    1    1
Verbal morphology|present progressive, A|PROG:A-AGR-FULL    1    1    0
Verbal morphology|present progressive, A|PROG:A-Gender-AGR    1    0    1

Verbal morphology|present progressive, DAT|PROG:NO-DAT-AGR    1    1    1
Verbal morphology|present progressive, DAT|PROG:DAT-AGR-FULL    1    1    0
Verbal morphology|present progressive, DAT|PROG:DAT-Gender-AGR    1    0    1

Verbal morphology|present progressive, O|PROG:NO-O-AGR    1    1    1
Verbal morphology|present progressive, O|PROG:O-AGR-FULL    1    1    0
Verbal morphology|present progressive, O|PROG:O-Gender-AGR    1    0    1

Verbal morphology|simple PAST, A|PST:NO-A-AGR    1    1    1
Verbal morphology|simple PAST, A|PST:A-AGR-FULL    1    1    0
Verbal morphology|simple PAST, A|PST:A-Gender-AGR    1    0    1

Nominal morphology|Nominal cases|<7 Cases    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1
Nominal morphology|Nominal cases|OBL-Cases    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
Nominal morphology|Nominal cases|DAT    0    0    0    0    0    0    0    0    1    1    1    1    1    1    1    1    0    0    0    0    0    0    0    0    1    1    1    1    1    1    1
Nominal morphology|Nominal cases|GEN    0    0    0    0    1    1    1    1    0    0    0    0    1    1    1    1    0    0    0    0    1    1    1    1    0    0    0    0    1    1    1
Nominal morphology|Nominal cases|O-case    0    0    1    1    0    0    1    1    0    0    1    1    0    0    1    1    0    0    1    1    0    0    1    1    0    0    1    1    0    0    1
Nominal morphology|Nominal cases|VOC    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0

Nominal morphology|Pronominal Cases|<7 Cases    0    0    0    0    0    0    0    1    1    1    1    1    1    1    1
Nominal morphology|Pronominal Cases|OBL-Cases    1    1    1    1    1    1    1    0    0    0    0    0    0    0    0
Nominal morphology|Pronominal Cases|A≠O    0    0    0    0    1    1    1    1    0    0    0    0    1    1    1
Nominal morphology|Pronominal Cases|DAT≠O    0    0    1    1    0    0    1    1    0    0    1    1    0    0    1
Nominal morphology|Pronominal Cases|VOC    0    1    0    1    0    1    0    1    0    1    0    1    0    1    0

Verbal morphology|simple PAST, DAT|PST:NO-DAT-AGR    1    1    1
Verbal morphology|simple PAST, DAT|PST:DAT-AGR-FULL    1    1    0
Verbal morphology|simple PAST, DAT|PST:DAT-Gender-AGR    1    0    1

Verbal morphology|simple PAST, O|PST:NO-O-AGR    1    1    1
Verbal morphology|simple PAST, O|PST:O-AGR-FULL    1    1    0
Verbal morphology|simple PAST, O|PST:O-Gender-AGR    1    0    1

Word order|Clitic pronouns finite verb|2nd position    1    1
Word order|Clitic pronouns finite verb|OV    1    0
Word order|Clitic pronouns finite verb|VO    0    1

Word order|Clitic pronouns infinitive|2nd position    1    1
Word order|Clitic pronouns infinitive|OV    1    0
Word order|Clitic pronouns infinitive|VO    0    1

Word order|Clitic pronouns participle|2nd position    1    1
Word order|Clitic pronouns participle|OV    1    0
Word order|Clitic pronouns participle|VO    0    1

Word order|Main clauses|SOV    0    1    1    1    0    1    0    0
Word order|Main clauses|SVO    0    1    0    1    1    0    1    0
Word order|Main clauses|V2    0    1    1    1    1    1    1    1
Word order|Main clauses|VSO    0    1    1    0    1    0    0    1

Word order|Subordinate clause|SOV    0    1    1    1    0    1    0    0
Word order|Subordinate clause|SVO    0    1    0    1    1    0    1    0
Word order|Subordinate clause|V2    0    1    1    1    1    1    1    1
Word order|Subordinate clause|VSO    0    1    1    0    1    0    0    1"""



featdep = featdep.split('\n\n')
for i in range(len(featdep)):
    featdep[i]=featdep[i].split('\n')


for i in range(len(featdep)):
    for j in range(len(featdep[i])):
        featdep[i][j]=featdep[i][j].split('    ')


banned = defaultdict(list)
for f in featdep:
    feattup = []
    for l in f:
        feattup.append(l[0])
    for i in range(1,len(f[0])):
        seq = []
        for l in f:
            seq.append(int(l[i]))
        banned[tuple(feattup)].append(tuple(seq))




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
    feats = random.sample(posterior.keys(),len(posterior.keys()))
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
        



bad = defaultdict(list)
for i in range(100):
    for n in pruneorders[0]:
        for s in banned.keys():
            counter = 0
            ftup = []
            for f in s:
                ftup.append(states[f][n][i])
            ftup = tuple(ftup)
            if ftup in banned[s]:
                counter += 1
            if counter != 0:
                bad[s].append(0)
            else:
                bad[s].append(1)


print 'Grid & Feature & Variant & Probability of licit combination\\\\'
for s in bad.keys():
    print '\\hline'
    for v in s[:1]:
        print ' & '.join(v.split('|')),'&','%.2f'%np.mean(bad[s]),'\\\\'
    for v in s[1:]:
        print ' & '.join(v.split('|')),'&','\\\\'
