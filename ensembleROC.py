#!/usr/bin/env python
from optparse import *
import os
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
from pylal import auxmvc_utils 
import bisect
import pickle

parser=OptionParser(usage="Generates summary plots for KW aux-triggers processed by StatPatternRecognition", version = "modified version by Young-Min Kim with original version, Kari Hodge's auxmvc_mvsc_results_plots.py in pylal of LSCSOFT")
parser.add_option("","--tag", help="filenames will be ROC_tag.png and efficiency_deadtime_tag.txt")
parser.add_option("","--output-dir", help="directory where output files will be written to")
(opts,files)=parser.parse_args()

try: os.mkdir(opts.output_dir)
except: pass


selected_runs = np.arange(100)		       # Among 100 runs
np.random.shuffle(selected_runs)	       # select randomly
selected_runs = np.sort(selected_runs[:20])    # 20 runs to get average

filename0 = "/data2/johnoh/Project/GANN/Aux_NMIFS35_100_Out"
filename1 = "/ALL_S6_959126400_hveto_channels_signif_dt_combined_norm_reduced_nmifs35_RR"
filename2 = "_n35n15n1_c10_mr08_p1000_g5_rmin-100.0_rmax100.0_gsigma1.0_w-01x01_y1001_f05g09_m6000.dat"

tag = opts.tag
for i in selected_runs:
  tag += "_Out"+str(i)

tmp = [auxmvc_utils.ReadMVSCTriggers([filename0+str(i)+filename1+str(j)+filename2 for j in range(10)]) for i in selected_runs]
data = tmp[0]

# The rank name is 'Bagger' for MVSC, 'glitch-rank' for ANN, and 'SVMRank' for SVM. The last column in *.dat files is ranking values, so the end component of first line in *.dat file is the rank name.
rank_name = data.dtype.names[-1]

data[rank_name] = np.average([tmp[i][rank_name] for i in range(len(tmp))],axis=0)

clean_data = data[np.nonzero(data['i']==0)[0],:]
glitch_data = data[np.nonzero(data['i']==1)[0],:]

all_ranks = np.concatenate((clean_data[rank_name],glitch_data[rank_name]))
all_ranks_sorted = np.sort(all_ranks)


FAP, TAP = auxmvc_utils.ROC(clean_data[rank_name], glitch_data[rank_name])
#FAP = false alarm percentage = number of random/clean times flagged as glitches
#TAP = true alarm percentage = number of glitches flagged as glitches

# Plot ROC curve
pl.figure(1)
pl.loglog(FAP,TAP, linewidth = 2.0)
pl.hold(True)
x = np.arange(min(TAP), max(TAP) + (max(TAP) - min(TAP))/1000.0, (max(TAP) - min(TAP))/1000.0)
pl.loglog(x,x, linestyle="dashed", linewidth = 2.0)
pl.xlabel('False Alarm Probability')
pl.ylabel('Efficiency')
pl.xlim([0,1])
pl.ylim([0,1])
pl.savefig(opts.output_dir + "/"+'ROC_'+tag+'.png')	
pl.close()

# save ROC curve in a file
roc_file = open(opts.output_dir + "/"+"ROC_" + tag + ".pickle", "w")
pickle.dump([FAP,TAP], roc_file)
roc_file.close()

#FAP is a list that is naturally sorted in reverse order (highest to lowest),
#we need to turn it into a regularly sorted list so that we can find the TAP for
#fiducial FAPs
FAP.sort()
edfile = open(opts.output_dir + "/"+'efficiency_deadtime_'+tag+'.txt','w')
for threshold in [min(FAP),.001,.01,.05,.1]:
	tmpindex=bisect.bisect_left(FAP,threshold)
	edfile.write("deadtime: "+str(FAP[tmpindex])+" efficiency: "+str(TAP[len(FAP)-tmpindex-1])+"\n")
edfile.close()
