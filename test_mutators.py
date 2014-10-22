import os
import sys
import numpy
from pyfann import libfann
from pyevolve import GenomeBase
from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators
import copy
from random import randint as rand_randint, gauss as rand_gauss, uniform as rand_uniform
from random import choice as rand_choice
from random import sample as rand_sample
from pyevolve import Consts
import G1DConnections
import GAnnMutators

#### Main #####

ann=libfann.neural_net()
#ann.create_from_file('../Iris-150_training_n4n8n1_c10_w-01_x01_y_12_f05g05_m1000.net')
train_data=libfann.training_data()
eval_data=libfann.training_data()
train_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_training.ann')
eval_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_evaluation.ann')

ann.create_sparse_array(1.0, [4,6,1])
ann.randomize_weights(-0.1,0.1)

ann_connec=ann.get_connection_array()
ann_connec=map(tuple,ann_connec)

num_neurons=ann.get_layer_array()
#ann_connec_changed=copy.copy(ann_connec)
genome = G1DConnections.G1DConnections(len(ann_connec))
genome.genomeList=numpy.array(ann_connec,dtype=[('from','i'),('to','i'),('weight','f')])
print "incoming nodes:"
print numpy.unique(genome['to'])


genome.setParams(rangemin=-100.0, rangemax=100.0, gauss_mu = 0.0, gauss_sigma = 100)
#print genome.getParams("rangemax",Consts.CdefRangeMax)

genome.mutator.set(GAnnMutators.G1DConnMutateNodes)
genome.evaluator.set(GAnnMutators.evaluateMSE)
#for gen in genome.getnode(genome['to'][4]):
#	print gen
	
#print GAnnMutators.getMutateConnIndex(genome,[genome[4]])

for gen in genome:
	print gen

ann_MSE= GAnnMutators.evaluateMSE(genome,network=ann,data=train_data)
print "initial MSE:%f" % ann_MSE

GAnnMutators.G1DConnMutateNodes(genome, mutation_nodes=[5,6], pmut=0.5)

for gen in genome:
	print gen

ann.set_weight_array(genome.toList())
ann_MSE= GAnnMutators.evaluateMSE(genome,network=ann,data=train_data)
print "mutated MSE:%f" % ann_MSE

GAnnMutators.G1DConnMutateNodes(genome, mutation_nodes=None, pmut=0.5)

for gen in genome:
	print gen

ann.set_weight_array(genome.toList())
ann_MSE= GAnnMutators.evaluateMSE(genome,network=ann,data=train_data)
print "50 percent randomly mutated. MSE:%f" % ann_MSE

#ga = GSimpleGA.GSimpleGA(genome)
#ga.setGenerations(1000)
#ga.evolve(freq_stats=100)

#print ga.bestIndividual()
#ann_connec_changed = ga.bestIndividual()

#ann.set_weight_array(ann_connec_changed)
#print "MSE:%f" % ann.test_data(train_data)

#ann.save('Iris-150_training.net')

