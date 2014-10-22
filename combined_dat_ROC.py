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
parser=OptionParser(usage="Combining NN .dat files and Making a combined ROC", version="1.0")
parser.add_option("-o","--output-dir", action="store", type="string", default="output", help="Output directory; Default is output")
parser.add_option("-i","--input-dir", action="store", type="string", default=".", help="Input directory; Default is .")
parser.add_option("-l", "--log-dir", action="store", type="string", default="log_dir", help="Log-file directory; Default is log_dir")
parser.add_option("-f", "--filename", action="store", type="string",default="none", help="Input file name")
parser.add_option("-N", "--n-of-RR", action="store", type="int", help="number of Round-robin")
parser.add_option("-T", "--user-tag", action="store", type="string", default="none", help="User defined tag")
############# ANN Parameter Options #####################
parser.add_option("-n","--neurons",action="store",type="string",help="number of neurons per each layer.usage: -n n5n4n3")
parser.add_option("-C","--connection-rate",action="store",type="float",default=1.0,help="connection rate. Default is 1 which gives full connections")
parser.add_option("-L","--learning-rate",action="store",type="float",default=0.7,help="connection rate. Default is 0.7 which gives full connections")
parser.add_option("-E","--max-epochs",action="store",type="int",default=1000,help="max iterations: default=1000")
parser.add_option("-w","--weights-min",action="store",type="float",default=-0.1,help="minimum weight: d=-0.1")
parser.add_option("-W","--weights-max",action="store",type="float",default=0.1,help="max weight: d=0.1")
parser.add_option("-y","--increase-factor",action="store",type="float",default=1.2,help="rprop increase factor. original default value of FANN library is 1.2")
parser.add_option("-z","--decrease-factor",action="store",type="float",default=0.5,help="rprop decrease factor:d=0.5")
parser.add_option("-d","--delta-min",action="store",type="float",default=0.0,help="rprop delta minimum:d=0.0")
parser.add_option("-D","--delta-max",action="store",type="float",default=50.0,help="rprop delta maximum:d=50")
parser.add_option("-H","--hidden-activation",action="store", type="string", default="SIGMOID", help="Activation fuction for hidden layers. Default is SIGMOIDSTEPWISE.")
parser.add_option("-O","--output-activation",action="store", type="string", default="SIGMOID", help="Activation fuction for output layer. Default is SIGMOID_STEPWISE.")
parser.add_option("-s","--steep-hidden",action="store",type="float",default=0.5,help="steepness of hidden layer activation function:d=0.5")
parser.add_option("-S","--steep-out",action="store",type="float",default=0.9,help="steepness of output layer activation function. original default value of FANN library is 0.9")
############### GA Parameter Options ######################
parser.add_option("-m","--mutation-rate",action="store",type="float",default=0.2,help="Generations for GA run:d=0.2")
parser.add_option("-g","--generations",action="store",type="int",default=20,help="Generations for GA run:d=20")
parser.add_option("-p","--population",action="store",type="int",default=20,help="Population Size for GA run:d=20")
parser.add_option("-r","--range-min",action="store",type="float",default=-1.0,help="minimum weight for GA run:d=-1.0")
parser.add_option("-R","--range-max",action="store",type="float",default=1.0,help="maximum weight for GA run:d=1.0")
parser.add_option("-u","--gauss-mu",action="store",type="float",default=0.0,help="mean value of Gaussian distribution for GA run:d=0.0")
parser.add_option("-U","--gauss-sigma",action="store",type="float",default=1.0,help="standard deviation of Gaussian distribution for GA run:d=1.0")

(opts,files)=parser.parse_args()

###### FIle i/o option ############
filename = opts.filename
tag= ''.join((filename.split('/')[-1]).split('RR')[:-1])
input_dir = opts.input_dir
output_dir = opts.output_dir
log_dir = opts.log_dir
n_of_RR = opts.n_of_RR
user_tag = opts.user_tag
##### ANN Parameter Option ########
neurons = opts.neurons
connection_rate=opts.connection_rate
learning_rate=opts.learning_rate
max_epochs = opts.max_epochs
min_weights = opts.weights_min
max_weights = opts.weights_max
increase_factor = opts.increase_factor
decrease_factor = opts.decrease_factor
min_delta = opts.delta_min
max_delta = opts.delta_max
hidden_activation = opts.hidden_activation
output_activation = opts.output_activation
steep_hidden=opts.steep_hidden
steep_out = opts.steep_out
###### GA Parameter Option ########
mutation_rate = opts.mutation_rate
generations = opts.generations
population = opts.population
min_range = opts.range_min
max_range = opts.range_max
gauss_mu = opts.gauss_mu
gauss_sigma = opts.gauss_sigma


input_tag = tag+'RR'
gann_tag = '_'+neurons+'_c'+''.join(str(connection_rate).split("."))+'_mr'+''.join(str(mutation_rate).split("."))+'_p'+str(population)+'_g'+str(generations)+'_rmin'+str(min_range)+'_rmax'+str(max_range)+'_gsigma'+str(gauss_sigma)+'_w'+''.join(str(min_weights).split("."))+'x'+''.join(str(max_weights).split("."))+'_y'+''.join(str(increase_factor).split("."))+'_f'+''.join(str(steep_hidden).split("."))+'g'+''.join(str(steep_out).split("."))+'_m'+str(max_epochs)
output_tag = tag+'combined'

print input_tag
print gann_tag
print output_tag

########## File Merger ##############
print 'Merging round-robin neural net dat files....'
p=0
g=open(output_dir+'/'+output_tag+gann_tag+'.dat','a')
for i in range(0,n_of_RR):
   f=open(input_dir+'/'+input_tag+str(i)+gann_tag+'.dat','r')
   if i==0:
       g.write(f.readline()[:-1])
       g.write('\n')
   else:
       f.readline()
       while 1:
           p+=1
           data=f.readline()
           if not data: break
           g.write(str(p))
           g.write(' ')
           g.write(' '.join(data.strip().split(' ')[1:]))
           g.write('\n')
   f.close()
g.close()

########### PlotROC function ##################
def PlotROC(Triggers,output_dir,tag,Title):
    clean_data = Triggers[np.nonzero(Triggers['i']==0)[0],:]
    glitch_data = Triggers[np.nonzero(Triggers['i']==1)[0],:]
    rank_name = Triggers.dtype.names[-1]
    all_ranks = np.concatenate((clean_data[rank_name],glitch_data[rank_name]))
    all_ranks_sorted = np.sort(all_ranks)
    FAP, TAP = auxmvc_utils.ROC(clean_data[rank_name],glitch_data[rank_name])
    #FAP = false alarm percentage = number of random/clean times flagged as glitches
    #TAP = true alarm percentage = number of glitches flagged as glitches

    # Plot ROC curve
    pylab.figure(1)
    pylab.plot(FAP,TAP, linewidth = 2.0)
    pylab.xscale('log')
    pylab.hold(True)
    x = np.arange(1e-5, 1, (1 - 1e-5)/1000)
    pylab.plot(x,x, linestyle="dashed", linewidth = 2.0)
    pylab.xlabel('False Alarm Probability')
    pylab.ylabel('Efficiency')
    pylab.xlim([1e-5,1])
    pylab.ylim([0,1])
    leg=pylab.legend(["ANN","Random Guess"], loc=0)
    leg_text=leg.get_texts()
    pylab.setp(leg_text,fontsize='large')
    pylab.title(Title,fontsize=12)
    #pylab.text(tag,horizontalalignment="center",verticalalignment="top")
    pylab.annotate("n"+tag.split("_n")[-1],(0.01,0.8),xycoords='axes fraction', fontsize=10)
    pylab.savefig(output_dir+"/"+"ROC"+tag+'.png',format='png')
    pylab.close()
    print "ROC curve is saved in \n%s" % output_dir+"/"+"ROC"+tag+'.png'

    # save ROC curve in a file
    roc_file = open(output_dir+"/"+"ROC"+tag+".pickle", "w")
    pickle.dump([FAP,TAP], roc_file)
    roc_file.close()

    #FAP is a list that is naturally sorted in reverse order (highest to lowest),
    #we need to turn it into a regularly sorted list so that we can find the TAP for
    #fiducial FAPs
    FAP.sort()
    edfile = open(output_dir+"/"+'efficiency_deadtime_'+tag+'.txt','w')
    for threshold in [min(FAP),.001,.01,.05,.1]:
        tmpindex=bisect.bisect_left(FAP,threshold)
        edfile.write("deadtime: "+str(FAP[tmpindex])+" efficiency: "+str(TAP[len(FAP)-tmpindex-1])+"\n")
    return True


########### Final Evaluation and Plotting a combined ROC ##############
Triggers=auxmvc_utils.ReadMVSCTriggers([output_dir+'/'+output_tag+gann_tag+'.dat'])
PlotROC(Triggers,output_dir,tag+'combined'+gann_tag,user_tag)
