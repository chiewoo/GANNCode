#!/usr/bin/env python

import numpy as np
from optparse import *

### Function of Returning mena value of 100 ranks ###
filename='testdata'
n_of_runs=5

def AvRanks(output_dir, filename, n_of_runs):
    ranks=[]
    for i in range(0, n_of_runs):
        fdata=np.loadtxt(output_dir+'/'+filename+str(i)+'.txt')
        rank=fdata.T[-1].tolist()
        ranks.append(rank)
        nranks=np.array(ranks)
    for i in range(len(nranks.T)):
        new_ranks=[]
        av_ranks=[]
        for j in range(0, n_of_runs):
            new_ranks.append(nranks[j][i].tolist())
            av_ranks.append(np.mean(new_ranks))
    nav_ranks=np.array(av_ranks)
    return nav_ranks
                                                                                                
print 'Generating temporal array files....'
for i in range(0, n_of_runs):
    f=open(filename+str(i)+'.txt','r')
    f.readline()
    g=open(filename+str(i)+'_temp.txt','a')
    p=0
    while 1:
        fdata=f.readline()
        if not fdata: break
        g.write(fdata)
    g.close()
    f.close()

### Computing Mean value of ranks ###
print 'Evaluating Mean Value of Ranks......'
n_ranks=AvRanks('.', filename, n_of_runs)
print n_ranks
g=open('AvRanked_'+filename,'a')
l=open(filename+str(i)+'_temp.txt','r')
print 'Write the result file....'
header=l.readline()
g.write(header)
while 1:
    p=0
    ldata=l.readline()
    if not ldata: break
    g.write(' '.join(ldata.split(' ')[:-1]))
    g.write(' ')
    g.write(str(n_ranks[p]))
    g.write('\n')
    p+=1
l.close()
g.close()
print 'All Jobs Done.'
