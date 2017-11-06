from __future__ import division
import re
import codecs
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt
import numpy as np
from numpy import log,exp
from numpy.random import normal,multivariate_normal,uniform


def genphy(): #read BEAST tree sample to extract parent/child relationships, branch lengths, phylogeographic information for each tree in sample
    text = []
    for l in open('diacl_merged_8_geo_year0_thinned.trees','r'):
        text.append(l.strip())
    tree0 = text.index([x for x in text if x.startswith('tree STATE_10010000')][0]) #line index for first tree in sample
    langdict = {}                                                            #key for languages in tree sample
    revlangdict = {}
    for l in text[text.index('Translate')+1:tree0-1]:
        langdict[codecs.decode(l.split()[1].strip(','),'utf8')]=l.split()[0]
        revlangdict[l.split()[0]] = codecs.decode(l.split()[1].strip(','),'utf8')
    branchmatrices = []                                                               #list for all sets of edges in each tree
    locs = []
    brlens = []
    parents = []
    children = []
#    burnin=len(text)-tree0
#    burnin=int(burnin/2)
#    for tree in text[tree0+burnin:-1:50]:  #thin tree sample by 10
    for tree in text[tree0:-1]: #if sample has already been burned-in and thinned
        tredge = []
        parse = re.split(r'([\[][^\]]+[\]]|,|\)[:;]+|\()',tree)                       #unappealing-looking regex to get tree sample into tractable form
        for i in range(len(parse))[::-1]:
            if parse[i] == '' or parse[i] == ',' or parse[i].startswith('[&rate'):
                parse.pop(i)
        for i in range(len(parse))[::-1]:
            if i in range(len(parse)-1):
                if parse[i+1].startswith('['):
                    if parse[i][-1] != '(' and parse[i][-1] != ')':
                        parse[i:i+2] = [''.join(parse[i:i+2])]
        for i in range(len(parse))[::-1]:
            if parse[i].endswith(':'):
                parse[i-1:i+2] = [''.join(parse[i-1:i+2])]
        j = len(langdict.keys())+len(langdict.keys())-1
        for i in range(len(parse))[::-1]:
            if parse[i].startswith('['):
                parse[i]=str(j)+parse[i]
                j -= 1
        parentlist=[str(len(langdict.keys())+len(langdict.keys())-1)]
        for i in range(len(parse)-1)[::-1]:
            if ':' in parse[i]:
                d = parse[i].split(':')[0]
                bl = parse[i].split(':')[1].strip(')')
                tredge.append([parentlist[-1],str(d),bl])
                if parse[i-1].endswith(')'):
                    parentlist.append(d)
            if parse[i] == '(':
                parentlist.pop(-1)
        loc = {}
        brlen = {}
        parent = {}
        child = defaultdict(list)
        rootloc = (re.split(r'\{|\}',tree)[-2].split(',')) #this is a really sloppy hack; find better way to get root location for BEAST tree sample
        loc['249'] = (float(rootloc[0]),float(rootloc[1]))
        for l in tredge:
#            print l
            p = l[0].split('[')[0]
            c = l[1].split('[')[0]
            s = re.split(r'[\{\}\,]',l[1])
            clon = float(s[1])
            clat = float(s[2])
            blen = float(l[2])
            loc[c] = (clon,clat)
#            loc[p] = (clon,clat)
            brlen[c] = blen/1000 #divide branch lengths by 1000 to avoid underflow in rate inference
            parent[c] = p
            child[p].append(c)
        locs.append(loc)
        brlens.append(brlen)
        parents.append(parent)
        children.append(child)
    for j in [7,3]:
        locs.pop(j)
        brlens.pop(j)
        parents.pop(j)
        children.pop(j)
    return langdict,revlangdict,locs,brlens,parents,children



#langdict,locs,brlens,parents,children=genphy()
