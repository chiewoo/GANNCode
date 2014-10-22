#!/usr/bin/env python
import numpy as np
import os
import glob
import pylab
import pickle
import bisect

from os import makedirs
from os.path import isdir, exists
from sys import exit
from optparse import *
from pylal import auxmvc_utils

############## File i/o Options #########################
parser=OptionParser(usage="Searching .dat files, Select random 20 samples and compute averaged ranks-making new .dat files: Making an averaged combined ROC", version="1.0")
parser.add_option("-o","--output-dir", action="store", type="string", default=".", help="Output directory; Default is output")
parser.add_option("-i","--input-dir", action="store", type="string", default=".", help="Input directory_tag; Default is .")
parser.add_option("-f", "--filename", action="store", type="string",default="none", help="Input file name")
parser.add_option("-n", "--n-of-runs", action="store", type="int", default="100",help="number of runs: default is 100")

(opts,files)=parser.parse_args()

###### FIle i/o option ############
filename = opts.filename
input_dir = opts.input_dir
output_dir = opts.output_dir
n_of_runs = opts.n_of_runs

if isdir(input_dir):
    print "Directory exists:", input_dir
else:
    print "Creating directory:", input_dir
    makedirs(input_dir)
if isdir(output_dir):
    print "Directory exists:", output_dir
else:
    print "Creating directory:", output_dir
    makedirs(output_dir)

### Function of Returning mena value of 100 ranks ###
print 'Generating temporal array files....'
for i in range(0, n_of_runs):
    f=open(input_dir+str(i)+'/'+filename,'r')
    f.readline()
    g=open(output_dir+'/'+filename+str(i)+'.txt','a')
    p=0
    while 1:
        fdata=f.readline()
        if not fdata: break
        g.write(fdata)
    g.close()
    f.close()

###
def AvRanks(output_dir, filename, n_of_runs):
    ranks=[]
    for i in range(0, n_of_runs):
        fdata=np.loadtxt(output_dir+'/'+filename+str(i)+'.txt')
        rank=fdata.T[-1].tolist()
        ranks.append(rank)
        nranks=np.array(ranks)
        print 'nranks:', nranks
    av_ranks=[]
    for i in range(len(nranks.T)):
        new_ranks=[]
        for j in range(0, n_of_runs):
            new_ranks.append(nranks[j][i].tolist())
        av_ranks.append(np.mean(new_ranks))
        print 'av_ranks:', av_ranks
    nav_ranks=np.array(av_ranks)
    return nav_ranks

### Computing Mean value of ranks ###
print 'Evaluating Mean Value of Ranks......'
n_ranks=AvRanks(output_dir,filename,n_of_runs)
print n_ranks
g=open(output_dir+'/'+'AvRanked_'+filename,'a')
l=open(input_dir+'0'+'/'+filename,'r')
print 'Write the result file....'
header=l.readline()
g.write(header)
p=0
while 1:
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

        
