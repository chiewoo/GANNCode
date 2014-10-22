#import Consts, Util
import GAnnConsts
#from FunctionSlot import FunctionSlot
#from Statistics import Statistics
from math import sqrt as math_sqrt
import logging
from pyevolve.GPopulation import GPopulation
from pyfann import libfann

try:
   from multiprocessing import cpu_count, Pool
   CPU_COUNT = cpu_count()
   MULTI_PROCESSING = True if CPU_COUNT > 1 else False
   logging.debug("You have %d CPU cores, so the multiprocessing state is %s", CPU_COUNT, MULTI_PROCESSING)
except ImportError:
   MULTI_PROCESSING = False
   logging.debug("You don't have multiprocessing support for your Python version !")


def key_raw_score(individual):
   """ A key function to return raw score

   :param individual: the individual instance
   :rtype: the individual raw score

   .. note:: this function is used by the max()/min() python functions

   """
   return individual.score

def key_fitness_score(individual):
   """ A key function to return fitness score, used by max()/min()

   :param individual: the individual instance
   :rtype: the individual fitness score

   .. note:: this function is used by the max()/min() python functions

   """
   return individual.fitness

def multiprocessing_eval(ind):
   """ Internal used by the multiprocessing """
   ann = libfann.neural_net()
   layers = GAnnConsts.CNetwork.get_layer_array()
   ann.create_sparse_array(GAnnConsts.CNetwork.get_connection_rate(), layers)
   ann.set_activation_function_hidden(GAnnConsts.CNetwork.get_activation_function(1,0))
   ann.set_activation_function_output(GAnnConsts.CNetwork.get_activation_function(len(layers)-1,0)) 
   ann.set_activation_steepness_hidden(GAnnConsts.CNetwork.get_activation_steepness(1,0))
   ann.set_activation_steepness_output(GAnnConsts.CNetwork.get_activation_steepness(len(layers)-1,0))
   ann.set_rprop_increase_factor(GAnnConsts.CNetwork.get_rprop_increase_factor())
   ann.set_rprop_decrease_factor(GAnnConsts.CNetwork.get_rprop_decrease_factor())
   ann.set_rprop_delta_min(GAnnConsts.CNetwork.get_rprop_delta_min())
   ann.set_rprop_delta_max(GAnnConsts.CNetwork.get_rprop_delta_max())
#   ann.print_parameters()
   
   ind.evaluate(network=ann, data=GAnnConsts.CTrainData)
   return ind.score

def multiprocessing_eval_full(ind):
   """ Internal used by the multiprocessing (full copy)"""
   ind.evaluate()
   return ind

class GAnnPopulation(GPopulation):
    def __init__(self, genome):
        GPopulation.__init__(self, genome)

    def evaluate(self, **args):
       """ Evaluate all individuals in population, calls the evaluate() method of individuals
    
       :param args: this params are passed to the evaluation function
 
       """
       # We have multiprocessing

       if self.multiProcessing[0] and MULTI_PROCESSING:
          logging.debug("Evaluating the population using the multiprocessing method")
          proc_pool = Pool()
 
          # Multiprocessing full_copy parameter
          if self.multiProcessing[1]:
             results = proc_pool.map(multiprocessing_eval_full, self.internalPop)
             for i in xrange(len(self.internalPop)):
                self.internalPop[i] = results[i]
          else:
#             GAnnConsts.CNetwork.print_parameters()
             results = proc_pool.map(multiprocessing_eval, self.internalPop)
             for individual, score in zip(self.internalPop, results):
                individual.score = score
       else:
          for ind in self.internalPop:
             ind.evaluate(**args)
 
       self.clearFlags()

    
