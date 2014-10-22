#!/usr/bin/env python

import G1DConnections
import GAnnMutators
import numpy as np
import random
from pyevolve import GSimpleGA, Consts
import GAnnGA
from pyfann import libfann

if __name__=='__main__':

    layers = [4,6,1]
    ndLayers = np.array(layers)
    bias = [1,1,0]
    ndBias = np.array(bias)

    ann=libfann.neural_net()
    ann.create_sparse_array(1.0, layers)
    ann.set_activation_function_hidden(libfann.SIGMOID)
    ann.set_activation_function_output(libfann.SIGMOID) 
    
    ann.set_activation_steepness_output(0.9)

    train_data=libfann.training_data()
    eval_data=libfann.training_data()
    train_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_training.ann')
    eval_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_evaluation.ann')
#    train_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_training.ann')
#    eval_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_evaluation.ann')

    #Consts.CCDefGAMutationRate=0.02
    genome = G1DConnections.G1DConnections()
    genome.evaluator.set(GAnnMutators.evaluateMSE)

    genome.setParams(rangemin=-1, rangemax=1, layers=layers, bias=bias, gauss_mu=0, gauss_sigma=1)
    ga = GAnnGA.GAnnGA(genome, ann, train_data)
    ga.setMutationRate(0.2)
    ga.setPopulationSize(20)
    ga.setGenerations(20)
    ga.evolve(freq_stats=1)

    best=ga.bestIndividual()
    ann.set_weight_array(best.toList())
    result=[ann.run(i) for i in eval_data.get_input()]
    expected_output = eval_data.get_output()
    import numpy as np
    x=(np.array(result).reshape(len(result)))
    y=(np.array(expected_output).reshape(len(expected_output)))

    from matplotlib import pylab as pl
    #sortIdx=x.argsort()
    #pl.plot(x[sortIdx], y[sortIdx], 'x')
    #pl.savefig('iris.png', format='png')
    #pl.close()
    pl.axis([0,30,-2,2])
    pl.plot(y, 'go')
    pl.plot(x,'rx',markersize=12)
    pl.xlabel('sample ID')
    pl.ylabel('class')
    leg=pl.legend(["Expected","GAnn"], loc=0)
    leg_text=leg.get_texts()
    pl.setp(leg_text,fontsize='large')
    pl.annotate(r'Iris Classification with GA+ANN',(0.30,1.05),xycoords='axes fraction', fontsize=14)
    pl.savefig('iris_classification.png', format='png')

    best=ga.bestIndividual()
    ann.set_weight_array(best.toList())
    print ann.test_data(train_data)
