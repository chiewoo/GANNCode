#!/usr/bin/env python
### This code uses FANN library 
### for Aritifical Neural Network analysis
### written by Young-Min Kim (young-min.kim@ligo.org)

###########################################
from optparse import *
import glob
import sys
import os
import numpy
import random
import time
###########################################
from pyfann import libfann
###########################################
import G1DConnections
import GAnnMutators
import GAnnEvaluators
from pyevolve import GSimpleGA, Consts
import GAnnGA
###########################################
from pylal import auxmvc_utils
from pylal import git_version
###########################################


parser=OptionParser(version=git_version.verbose_msg)
parser.add_option("-t","--training-file",action="store",type="string",help="training data file")
parser.add_option("","--output-dir",action="store",type="string",default=".",help="saving results")
parser.add_option("-s","--saving-results",action="store",type="string",default=False,help="saving result network")
############### network configuration parameters######################################
parser.add_option("-n","--neurons",action="store",type="string",help="number of neurons per each layer. 5,4,3 --> layer 1,2,3 have 5,4,3 neurons, respectively.First number is for input layer and last number is for output layer. ")
parser.add_option("-c","--connection-rate",action="store",type="float",default=1.0,help="connection rate. Default is 1 which gives full connections")
parser.add_option("-d","--hidden-activation",action="store", type="string", default="SIGMOID", help="Activation fuction for hidden layers. Default is SIGMOID.")
parser.add_option("-o","--output-activation",action="store", type="string", default="SIGMOID", help="Activation fuction for output layer. Default is SIGMOID.")
parser.add_option("-f","--steep-hidden",action="store",type="float",default=0.5,help="steepness of hidden layer activation function")
parser.add_option("-g","--steep-out",action="store",type="float",default=0.5,help="steepness of output layer activation function")
############### FANN parameters######################################
parser.add_option("","--learning-rate",action="store",type="float",default=0.7,help="connection rate. Default is 1 which gives full connections")
parser.add_option("-m","--max-epochs",action="store",type="int",default=1000,help="max iterations")
parser.add_option("-w","--weights-min",action="store",type="float",default=-0.1,help="minimum weight")
parser.add_option("-x","--weights-max",action="store",type="float",default=0.1,help="max weight")
parser.add_option("-y","--increase-factor",action="store",type="float",default=1.2,help="rprop increase factor")
parser.add_option("-z","--decrease-factor",action="store",type="float",default=0.5,help="rprop decrease factor")
parser.add_option("-a","--delta-min",action="store",type="float",default=0.0,help="rprop delta minimum")
parser.add_option("-b","--delta-max",action="store",type="float",default=50.0,help="rprop delta maximum")
############### GA parameters######################################
parser.add_option("","--mutation-rate",action="store",type="float",default=0.2,help="Generations for GA run")
parser.add_option("","--generations",action="store",type="int",default=20,help="Generations for GA run")
parser.add_option("","--population",action="store",type="int",default=20,help="Population Size for GA run")
parser.add_option("","--range-min",action="store",type="float",default=-1.0,help="minimum weight for GA run")
parser.add_option("","--range-max",action="store",type="float",default=1.0,help="maximum weight for GA run")
parser.add_option("","--gauss-mu",action="store",type="float",default=0.0,help="mean value of Gaussian distribution for GA run")
parser.add_option("","--gauss-sigma",action="store",type="float",default=1.0,help="standard deviation of Gaussian distribution for GA run")
(opts,files)=parser.parse_args()

############# MAIN ###########################
start_time=time.time()
try: os.mkdir(opts.output_dir)
except: pass

########## Network Configuration ###########
print "Creating network."

# define paramters for total layers : [num_neurons[0],num_neurons[1],.....,num_neurons[-1]]. num_neurons[0] is the number of input variables and num_neurons[-1] is the number of output variables.
layers = map(int,opts.neurons.split(","))
ndLayers = numpy.array(layers)
bias = [1]+[1 for i in range(len(layers[1:-1]))]+[0]
ndBias = numpy.array(bias)
print "The layer structure is following:"
print layers

# generate neural network
ann = libfann.neural_net()
ann.create_sparse_array(opts.connection_rate, layers)

print "Setting newtork parameters"
# activation functions : SIGMOID,SIGMOID_STEPWISE,SIGMOID_SYMMETRIC,SIGMOID_SYMMETRIC_STEPWISE,LINEAR,THERESHOLD,THRESHOLD_SYMMETRIC,GAUSSIAN,GAUSSIAN_SYMMETRIC,ELLIOT,ELLIOT_SYMMETRIC,LINEAR_PIECE,LINEAR_PIECE_SYMMETRIC,SIN_SYMMETRIC,COS_SYMMETRIC,SIN,COS
if opts.hidden_activation == "SIGMOID_SYMMETRIC_STEPWISE":
	ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
elif opts.hidden_activation == "GAUSSIAN":
	ann.set_activation_function_hidden(libfann.GAUSSIAN)
elif opts.hidden_activation == "GAUSSSIAN_SYMMETRIC":
	ann.set_activation_function_hidden(libfann.GAUSSIAN_SYMMETRIC)
elif opts.hidden_activation == "SIGMOID":
	ann.set_activation_function_hidden(libfann.SIGMOID)
else:
	ann.set_activation_function_hidden(libfann.SIGMOID_STEPWISE)
ann.set_activation_steepness_hidden(opts.steep_hidden)

if opts.output_activation == "SIGMOID_SYMMETRIC_STEPWISE":
	ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
elif opts.output_activation == "GAUSSIAN":
	ann.set_activation_function_output(libfann.GAUSSIAN)
elif opts.output_activation == "GAUSSIAN_SYMMETRIC":
	ann.set_activation_function_output(libfann.GAUSSIAN_SYMMETRIC)
elif opts.output_activation == "SIGMOID":
	ann.set_activation_function_output(libfann.SIGMOID)
else:
	ann.set_activation_function_output(libfann.SIGMOID_STEPWISE)
ann.set_activation_steepness_output(opts.steep_out)


########## Import training data #####################
print "Getting training data : %s" % opts.training_file
train_data = libfann.training_data()
train_data.read_train_from_file(opts.training_file.replace(".pat",".ann"))
#train_data.scale_train_data(0.0,1.0)

########## GA Training #####################
print "Setting GA training parameters"
genome = G1DConnections.G1DConnections()
genome.evaluator.set(GAnnEvaluators.evaluateMSE)

genome.setParams(rangemin=opts.range_min, rangemax=opts.range_max, layers=layers, bias=bias, gauss_mu=opts.gauss_mu, gauss_sigma=opts.gauss_sigma)
#genome.mutator.set(GAnnMutators.G1DConnMutateNodes)
ga = GAnnGA.GAnnGA(genome, ann, train_data)
ga.setMutationRate(opts.mutation_rate)
ga.setPopulationSize(opts.population)
ga.setGenerations(opts.generations)
print "Start running GA"
ga.evolve(freq_stats=1)
print "GA MSE : %f" % ann.test_data(train_data)

# choose best initial connection weights
best=ga.bestIndividual()
# set initial connection weights on ann
ann.set_weight_array(best.toList())
print "Best initial connection wetighs are set on Network"

########## FANN Training #####################
print "Setting FANN training parameters"
learning_rate = opts.learning_rate
desired_error = 0.001
max_iterations = opts.max_epochs
iterations_between_reports = 100

#ann.set_input_scaling_params(train_data,-100,100)

# training algorithm : INCREMENTAL, BATCH, RPROP, QUICKPROP
# learning rate and learning momentum are not used during RPROP training
ann.set_training_algorithm(libfann.TRAIN_RPROP)
ann.set_learning_rate(learning_rate)
#ann.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
#ann.set_learning_momentum(learning_momentum)

#rpop parameters
ann.set_rprop_increase_factor(opts.increase_factor)# >1, default 1.2
ann.set_rprop_decrease_factor(opts.decrease_factor)# <1, default 0.5
ann.set_rprop_delta_min(opts.delta_min)# small positive number, default 0.0
ann.set_rprop_delta_max(opts.delta_max)# positive number, default 50.0

#ann.init_weights(train_data)
#ann.randomize_weights(opts.weights_min,opts.weights_max)
# initial connection weights for FANN  

ann.print_parameters()
#ann.print_connections()

print "Start training network with %s-algorithm in FANN" % ann.get_training_algorithm()
start_fann_time = time.time()
ann.train_on_data(train_data, max_iterations, iterations_between_reports, desired_error)
end_fann_time = time.time()
print "Time elpased for FANN training: %f seconds" % (end_fann_time - start_fann_time)
ann.save(opts.output_dir+"/"+opts.saving_results)
print "Trained Network by GA+FANN is saved in \n%s/%s." % (opts.output_dir,opts.saving_results)
end_time = time.time()
print "Total Running Time: %f seconds" % (end_time - start_time)
