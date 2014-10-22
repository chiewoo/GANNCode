#!/usr/bin/env python

import os
import G1DConnections
import GAnnMutators
import GAnnCrossovers
import GAnnInitializators
import GAnnUtils
import GAnnConsts
import GAnnEvaluators
import GAnnSelectors
import exceptions
import numpy as np
import random
from pyevolve import GSimpleGA, Consts, Util
import GAnnGA
from pyfann import libfann
import time
from pylal import auxmvc_utils
from optparse import *
import matplotlib
matplotlib.use('Agg')
import pylab
import pickle
import bisect

def Evaluation(network,eval_data,output_dir,tag):
    result_file=open(output_dir+"/"+tag+".dat",'w')
    inputs=eval_data.get_input()
    outputs=eval_data.get_output()
    w_row = np.ones(len(inputs))
    attributes = ['x'+str(k) for k in range(len(inputs[0]))] # this line will be corrected as real attributes of data
    variables=['index','i','w']+attributes+['rank']
    result_file.write(' '.join(variables)+"\n")
    formats = ['i','i']+['g8' for l in range(len(variables)-2)]
    for i in range(len(inputs)):
        results=network.run(inputs[i])
        variable_line = " ".join([str(var) for var in inputs[i]])
        result_file.write(str(i+1)+" "+str(outputs[i][0])+" "+str(w_row[i])+" "+variable_line+" "+str(results[0])+"\n")
    result_file.close()
    print "Evaluation results are saved in \n%s" % output_dir+"/"+tag+".dat"
    Triggers = auxmvc_utils.ReadMVSCTriggers([output_dir+"/"+tag+".dat"])

    return Triggers

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
    pylab.savefig(output_dir+"/"+"ROC_"+tag+'.png',format='png')
    pylab.close()
    print "ROC curve is saved in \n%s" % output_dir+"/"+"ROC_"+tag+'.png' 

    # save ROC curve in a file
    roc_file = open(output_dir+"/"+"ROC_"+tag+".pickle", "w")
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

def PlotScatter(ranks,expected_ranks,output_dir,tag,Title):
    sortIdx=ranks.argsort()
    pylab.axis([0,1000,-1,2])
    pylab.plot(expected_ranks, 'go')
    pylab.plot(ranks,'rx',markersize=12)
    pylab.xlabel('sample ID')
    pylab.ylabel('class')
    leg=pylab.legend(["Expected","GAnn"], loc=0)
    leg_text=leg.get_texts()
    pylab.setp(leg_text,fontsize='large')
    pylab.title(Title,fontsize=12)
#pylab.text(tag,horizontalalignment="center",verticalalignment="top")
    pylab.annotate("n"+tag.split("_n")[-1],(0.01,0.8),xycoords='axes fraction', fontsize=10)
    pylab.savefig(output_dir+"/"+'Hist_'+tag+'.png', format='png')
    pylab.close()

    return True

parser=OptionParser(usage="Train & Evaluation with GA and FANN",version="NIMS version for GA+FANN running")
parser.add_option("-t","--training-file",action="store",type="string",help="training data file")
parser.add_option("-e","--evaluation-file",action="store",type="string",help="training data file")
parser.add_option("","--output-dir",action="store",type="string",default=".",help="saving results")
parser.add_option("","--tag",action="store",type="string",default=False,help="Name tag will be put on file names.")
parser.add_option("-n","--neurons",action="store",type="string",help="number of neurons per each layer. 5,4,3 --> layer 1,2,3 have 5,4,3 neurons, respectively.First number is for input layer and last number is for output layer. ")
parser.add_option("-c","--connection-rate",action="store",type="float",default=1.0,help="connection rate. Default is 1 which gives full connections")
parser.add_option("","--learning-rate",action="store",type="float",default=0.7,help="connection rate. Default is 0.7 which gives full connections")
parser.add_option("-m","--max-epochs",action="store",type="int",default=1000,help="max iterations: default=1000")
parser.add_option("-w","--weights-min",action="store",type="float",default=-0.1,help="minimum weight: d=-0.1")
parser.add_option("-x","--weights-max",action="store",type="float",default=0.1,help="max weight: d=0.1")
parser.add_option("-y","--increase-factor",action="store",type="float",default=1.2,help="rprop increase factor. original default value of FANN library is 1.2")
parser.add_option("-z","--decrease-factor",action="store",type="float",default=0.5,help="rprop decrease factor:d=0.5")
parser.add_option("-a","--delta-min",action="store",type="float",default=0.0,help="rprop delta minimum:d=0.0")
parser.add_option("-b","--delta-max",action="store",type="float",default=50.0,help="rprop delta maximum:d=50")
parser.add_option("-d","--hidden-activation",action="store", type="string", default="SIGMOID", help="Activation fuction for hidden layers. Default is SIGMOID_STEPWISE.")
parser.add_option("-o","--output-activation",action="store", type="string", default="SIGMOID", help="Activation fuction for output layer. Default is SIGMOID_STEPWISE.")
parser.add_option("-f","--steep-hidden",action="store",type="float",default=0.5,help="steepness of hidden layer activation function:d=0.5")
parser.add_option("-g","--steep-out",action="store",type="float",default=0.9,help="steepness of output layer activation function. original default value of FANN library is 0.9")
parser.add_option("","--mutation-rate",action="store",type="float",default=0.2,help="Generations for GA run:d=0.2")
parser.add_option("","--generations",action="store",type="int",default=20,help="Generations for GA run:d=20")
parser.add_option("","--population",action="store",type="int",default=20,help="Population Size for GA run:d=20")
parser.add_option("","--range-min",action="store",type="float",default=-1.0,help="minimum weight for GA run:d=-1.0")
parser.add_option("","--range-max",action="store",type="float",default=1.0,help="maximum weight for GA run:d=1.0")
parser.add_option("","--gauss-mu",action="store",type="float",default=0.0,help="mean value of Gaussian distribution for GA run:d=0.0")
parser.add_option("","--gauss-sigma",action="store",type="float",default=1.0,help="standard deviation of Gaussian distribution for GA run:d=1.0")
parser.add_option("-I","--initializator",action="store",type="string",default="uniform",help="Initializator for initial population of GA:d=uniform")
parser.add_option("","--import-network",action="store",type="string",default="",help="Import network from existing file")
parser.add_option("-G","--GA-only",action="store_true",default=False,help="Run only GA part. Cannot be given with -F/--FANN-only.")
parser.add_option("-F","--FANN-only",action="store_true",default=False,help="*NOTE that not implemented yet*. Run only FANN part. Cannot be given with -G/--GA-only.")
parser.add_option("-M","--enable_multiprocess",action="store_true",default=False,help="Enable multiprocess.")

(opts,files)=parser.parse_args()

if (opts.GA_only and opts.FANN_only):
    Util.raiseException("-F/--FANN-only and -G/--GA-only cannot be given together.")

if opts.initializator == "gauss":
    gaInit=GAnnInitializators.G1DConnInitializatorGaussian
elif opts.initializator == "uniform":
    gaInit=GAnnInitializators.G1DConnInitializatorUniform
else:
    raise exceptions.ValueError("Wrong argument {0} for {1} ".format(opts.initializator, '--initializator'))

if __name__=='__main__':

    try: os.mkdir(opts.output_dir)
    except: pass
    start_time=time.time()    
    ann=libfann.neural_net()
    if opts.import_network:
        print "%% Import from existing network."
        print "%% The FANN parameters except max-epochs will be overwritten with the imported values!"
        ann.create_from_file(opts.import_network)
        layers = ann.get_layer_array()
        bias = ann.get_bias_array()
        increase_factor = ann.get_rprop_increase_factor()
        decrease_factor = ann.get_rprop_decrease_factor()
        steep_hidden = ann.get_activation_steepness(1,0)
        steep_out = ann.get_activation_steepness(len(layers)-1,0)
        delta_min = ann.get_rprop_delta_min()
        delta_max = ann.get_rprop_delta_max()
        connection_rate = ann.get_connection_rate()
        
    else:
        layers = map(int,opts.neurons.split(","))
        bias = [1]+[1 for i in range(len(layers[1:-1]))]+[0]
        increase_factor = opts.increase_factor
        decrease_factor = opts.decrease_factor
        steep_hidden = opts.steep_hidden
        steep_out = opts.steep_out
        delta_min = opts.delta_min
        delta_max = opts.delta_max
        connection_rate = opts.connection_rate
        
    ndLayers = np.array(layers)
    ndBias = np.array(bias)

    # setting network parameters when not importing existing network.
    if not opts.import_network:
        ann.create_sparse_array(connection_rate, layers)
        ann.set_activation_function_hidden(libfann.SIGMOID)
        ann.set_activation_function_output(libfann.SIGMOID) 
        ann.set_activation_steepness_hidden(steep_hidden)
        ann.set_activation_steepness_output(steep_out)

# setting training/evaluation data
    train_data=libfann.training_data()
    eval_data=libfann.training_data()
    train_data.read_train_from_file(opts.training_file)
    train_data.shuffle_train_data()
    eval_data.read_train_from_file(opts.evaluation_file)
#    train_data.read_train_from_file('/data2/youngmin/Projects/AuxMVC/S6/week_959131741/data/100ms_window_data/normalized_data/959126400_hveto_channels_signif_dt_data_log/ALL_S6_959126400_hveto_channels_signif_dt_set_0_training.ann')
#    eval_data.read_train_from_file('/data2/youngmin/Projects/AuxMVC/S6/week_959131741/data/100ms_window_data/normalized_data/959126400_hveto_channels_signif_dt_data_log/ALL_S6_959126400_hveto_channels_signif_dt_set_0_evaluation.ann')
#    train_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_training.ann')
#    eval_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_evaluation.ann')

# setting GA parameters
    #Consts.CCDefGAMutationRate=0.02
    if opts.import_network:
        totalConnections = ann.get_total_connections()
        genome = G1DConnections.G1DConnections(totalConnections)
        genome.genomeList = GAnnUtils.ToGAnnConn(ann.get_connection_array())
    else:
        totalConnections = int(np.sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)]))
        genome = G1DConnections.G1DConnections(totalConnections)
        
    genome.evaluator.set(GAnnEvaluators.evaluateMSE)

    genome.setParams(rangemin=opts.range_min, rangemax=opts.range_max, layers=layers, bias=bias, gauss_mu=opts.gauss_mu, gauss_sigma=opts.gauss_sigma)
#    genome.mutator.set(GAnnMutators.G1DConnMutateNodes)
    genome.initializator.set(gaInit)
    genome.mutator.set(GAnnMutators.G1DConnUnbiasedMutateWeights)
    genome.crossover.set(GAnnCrossovers.G1DConnCrossoverWeights)
    
    GAnnConsts.CTrainData = train_data
    GAnnConsts.CNetwork = ann
    ga = GAnnGA.GAnnGA(genome, ann, train_data)
    ga.selector.set(GAnnSelectors.GRouletteWheel)
    ga.setMutationRate(opts.mutation_rate)
    ga.setPopulationSize(opts.population)
    ga.setGenerations(opts.generations)
    if opts.enable_multiprocess:
        ga.setMultiProcessing()

    print "Start running GA"
    print "Training File : \n%s" % opts.training_file
    ga.evolve(freq_stats=1)

    best=ga.bestIndividual()
    ann.set_weight_array(best.toList())
    print "GA MSE : %f" % ann.test_data(train_data)
    
    ga_tag=opts.tag+'_n'+'n'.join(map(str,layers))+'_c'+''.join(str(connection_rate).split("."))+'_mr'+''.join(str(opts.mutation_rate).split("."))+'_p'+str(opts.population)+'_g'+str(opts.generations)+'_rmin'+str(opts.range_min)+'_rmax'+str(opts.range_max)+'_gsigma'+str(opts.gauss_sigma)
    ann.save(opts.output_dir+"/"+ga_tag+'.net')
    print "Trained Network by GA is saved in \n%s" % opts.output_dir+"/"+ga_tag+'.net'

    print "Start Evaluation"
    print "Evaluation File : \n%s" % opts.evaluation_file
    ga_results=Evaluation(ann,eval_data,opts.output_dir,ga_tag)
    PlotROC(ga_results,opts.output_dir,ga_tag,"ROC of GA:"+opts.evaluation_file.split("/")[-1])
    PlotScatter(ga_results[ga_results.dtype.names[-1]],ga_results['i'],opts.output_dir,ga_tag,"Scatter Plot:"+opts.evaluation_file.split("/")[-1])

    if not opts.GA_only:
        # setting rprop parameters in FANN
        max_iterations=opts.max_epochs
        iterations_btw_reports=100
        desired_error=0.00001
        #setting RPROP parameters
        
        ann.set_rprop_increase_factor(increase_factor) # >1, default 1.2
        ann.set_rprop_decrease_factor(decrease_factor) # <1, default 0.5
        ann.set_rprop_delta_min(delta_min) # small positive number, default 0.0
        ann.set_rprop_delta_max(delta_max) # positive number, default 50.0
        # RPROP training 
        print "Start running RPROP training"
        print "Training File : \n%s" % opts.training_file
        start_fann_time = time.time()
        ann.train_on_data(train_data, max_iterations, iterations_btw_reports, desired_error)
        end_fann_time = time.time()
        print "Time elpased for FANN training: %f seconds" % (end_fann_time - start_fann_time)
        
        ann_tag=ga_tag+'_w'+''.join(str(opts.weights_min).split("."))+'x'+''.join(str(opts.weights_max).split("."))+'_y'+''.join(str(increase_factor).split("."))+'_f'+''.join(str(steep_hidden).split("."))+'g'+''.join(str(steep_out).split("."))+'_m'+str(opts.max_epochs)
        ann.save(opts.output_dir+"/"+ann_tag+'.net')
        print "Trained Network by GA+FANN is saved in \n%s" % opts.output_dir+"/"+ann_tag+'.net'
        print "Start Evaluation"
        print "Evaluation File : \n%s" % opts.evaluation_file
        ann_results=Evaluation(ann,eval_data,opts.output_dir,ann_tag)
        PlotROC(ann_results, opts.output_dir,ann_tag,"ROC of GA+ANN:"+opts.evaluation_file.split("/")[-1])
        PlotScatter(ann_results[ann_results.dtype.names[-1]],ann_results['i'],opts.output_dir,ann_tag,"Scatter Plot:"+opts.evaluation_file.split("/")[-1])
    
    end_time = time.time()
    print "Total Running Time: %f seconds" % (end_time - start_time)
    
