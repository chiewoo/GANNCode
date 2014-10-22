#!/usr/bin/env python

import G1DConnections
import GAnnMutators
import numpy as np
import random
from pyevolve import GSimpleGA, Consts
import GAnnGA
from pyfann import libfann
import time

if __name__=='__main__':

    start_time=time.time()
    layers = [70,10,1]
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
    train_data.read_train_from_file('/data2/youngmin/Projects/AuxMVC/S6/week_959131741/data/100ms_window_data/normalized_data/959126400_hveto_channels_signif_dt_data_log/ALL_S6_959126400_hveto_channels_signif_dt_set_0_training.ann')
    eval_data.read_train_from_file('/data2/youngmin/Projects/AuxMVC/S6/week_959131741/data/100ms_window_data/normalized_data/959126400_hveto_channels_signif_dt_data_log/ALL_S6_959126400_hveto_channels_signif_dt_set_0_evaluation.ann')
#    train_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_training.ann')
#    eval_data.read_train_from_file('/home/youngmin/Projects/GANN/data/Iris-150_evaluation.ann')

    #Consts.CCDefGAMutationRate=0.02
    genome = G1DConnections.G1DConnections()
    genome.evaluator.set(GAnnMutators.evaluateMSE)

    genome.setParams(rangemin=-1, rangemax=1, layers=layers, bias=bias, gauss_mu=0, gauss_sigma=10)
    ga = GAnnGA.GAnnGA(genome, ann, train_data)
    ga.setMutationRate(0.5)
    ga.setPopulationSize(500)
    ga.setGenerations(20)
    ga.evolve(freq_stats=1)

    best=ga.bestIndividual()
    ann.set_weight_array(best.toList())
    print ann.test_data(train_data)

    ann.save('s6_ga_p50g20_range1sigma1.net')
    max_iterations=4000
    iterations_btw_reports=100
    desired_error=0.001
    #setting RPROP parameters
    ann.set_rprop_increase_factor(1.3) # >1, default 1.2
    ann.set_rprop_decrease_factor(0.1) # <1, default 0.5
    ann.set_rprop_delta_min(0.0) # small positive number, default 0.0
    ann.set_rprop_delta_max(50.0) # positive number, default 50.0
    # RPROP training 
    start_fann_time = time.time()
    ann.train_on_data(train_data, max_iterations, iterations_btw_reports, desired_error)
    end_fann_time = time.time()
    print "Time elpased for fann training: %f seconds" % (end_fann_time - start_fann_time)
    ann.save('s6_gafann_p50g20_range1sigma1.net')
    result=[ann.run(i) for i in eval_data.get_input()]
    expected_output = eval_data.get_output()
    import numpy as np
    x=(np.array(result).reshape(len(result)))
    y=(np.array(expected_output).reshape(len(expected_output)))

    from matplotlib import pylab as pl
    sortIdx=x.argsort()
    pl.axis([-0.5,1.5,-0.5,1.5])
    pl.plot(x[sortIdx], y[sortIdx], 'x')
    pl.savefig('s6_x_fann_p50g20_range1sigma1.png', format='png')
    pl.close()
    pl.axis([0,600,-1,2])
    pl.plot(y, 'go')
    pl.plot(x,'rx',markersize=12)
    pl.xlabel('sample ID')
    pl.ylabel('class')
    leg=pl.legend(["Expected","GAnn"], loc=0)
    leg_text=leg.get_texts()
    pl.setp(leg_text,fontsize='large')
    pl.annotate('S6_959126400 Classification with GA+ANN',(0.30,1.05),xycoords='axes fraction', fontsize=14)
    pl.savefig('s6_gafann_p50g20_range1sigma1.png', format='png')
    end_time = time.time()
    print "Total Running Time: %f seconds" % (end_time - start_time)
#    best=ga.bestIndividual()
#    ann.set_weight_array(best.toList())
#    print ann.test_data(train_data)
