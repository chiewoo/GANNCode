#!/usr/bin/env python

from optparse import *
import matplotlib
matplotlib.use('Agg')
import pylab
import pickle
import numpy
import os

parser=OptionParser(usage="supply several pickle files with false alarm percentages v. true alarm percentages, this code will plot them all on the same image", version="modified version by Young-Min Kim with original version, Kari Hodge's auxmvc_ROC_combiner.py in pylal of LSCSOFT")
parser.add_option("","--tag", help="filenames will be combined_ROC_tag.png")
parser.add_option("","--labels", default=False, help="labels for each plot")
parser.add_option("","--title", default=False, help="title")
parser.add_option("","--output-dir", default=".", help="directory where output files will be written to")

(opts,picklefiles)=parser.parse_args()

try: os.mkdir(opts.output_dir)
except: pass

labels=[]

fig = pylab.figure(1)
ax = pylab.axes()
matplotlib.rcParams.update({'legend.fontsize': 8})
ax.set_color_cycle(['b','g','r','y',(0.9,0.5,0.2),'c','m',(0.5,0.,0.8),'k'])

#full_data=pickle.load(open('/home/youngmin/Projects/AuxMVC/S6/week_959131741/ANN/reference_data/ROC_ALL_S6_full_n1250n10n10n10n1_y1001_f05g09_m2000.pickle'))
#pylab.loglog(full_data[0],full_data[1])
#pylab.plot(full_data[0],full_data[1])
#pylab.xscale('log')
#labels.append('S6_full, 5s window')

#dt_data=pickle.load(open('/home/youngmin/Projects/AuxMVC/S6/week_959131741/ANN/reference_data/ROC_ALL_S6_dt_n250n10n10n10n1_y1001_f05g09_m2000.pickle'))
#pylab.plot(dt_data[0],dt_data[1])
#pylab.xscale('log')
#labels.append('S6_dt, 5s window')

for file in picklefiles:
	data = pickle.load(open(file))
	#thislabel=file.split('/')[-1].split('.')[0]
	#dqlabel=thislabel.split('_')[1]
	thislabel=file.split('ALL_')[-1].split('_n')[0]
	labels.append(thislabel+", 100ms window")
	#pylab.loglog(data[0],data[1])
	pylab.plot(data[0],data[1])
	pylab.xscale('log')
	#pylab.yscale('log')
	pylab.xlabel('False Alarm Probability')
	pylab.ylabel('Efficiency')
	pylab.xlim([0.0001,1])
	pylab.ylim([0,1])
	pylab.hold(True)


x = numpy.arange(0.0001,1.0,0.001)
pylab.plot(x,x, linestyle="dashed", color='grey', linewidth = 2.0)
labels.append("Random guess")
pylab.xlim([0.0001,1])
pylab.ylim([0,1])
pylab.grid("on")

if opts.labels:
	labels=opts.labels.split(",")

prop=matplotlib.font_manager.FontProperties(size=6.5)
#pylab.legend(labels,loc=0,prop=prop)
#pylab.text(0.01,0.9,opts.tag,horizontalalignment='center',fontsize=16)
if opts.title:
	pylab.title(opts.title,fontsize=12)
else:
	pylab.title(opts.tag+": Narrowing dt window",fontsize=12)
pylab.savefig(opts.output_dir+'/combined_ROC_'+opts.tag+'.png')

