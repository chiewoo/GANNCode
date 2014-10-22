"""

:mod:`AnnCrossovers` -- crossover methods module for FANN
=====================================================================
########################
#
# Developers: Kyungmin Kim, Young-Min Kim, Jaehyun Lee, John J. Oh, Sang Hoon Oh, Edwin J. Son
# First Developed: July 16 2012 (Green Wise Forest)
# Special Thanks to: Hyun Kyu Lee and Hyung Mok Lee
# This module can be used in the weight selection and node selection of crossover operation
# in the genetic algorithm (GA) with artificial neural networks (ANN). 
#
#########################

"""


from random import randint as rand_randint, choice as rand_choice
from random import random as rand_random
import math
import numpy as np
from pyevolve import Util
from pyevolve import Consts
#import G1DConnections

##################################
##    G1DConnCrossoverWeights   ##
##################################

def G1DConnCrossoverWeights(genome, **args):
    sister = None
    brother = None
    gMom = args["mom"]
    gDad = args["dad"]

    sister = gMom.clone()
    brother = gDad.clone()
    sister.resetStats()
    brother.resetStats()

    for i in xrange(len(gMom)):
      if Util.randomFlipCoin(Consts.CDefG1DListCrossUniformProb):
            temp = sister[i][2]
            sister[i][2] = brother[i][2]
            brother[i][2] = temp
            
    return (sister, brother)


###############################
##   G1DConnCrossoverNodes   ##
###############################

def G1DConnCrossoverNodes(genome, **args):
    sister = None
    brother = None
    gMom = args["mom"]
    gDad = args["dad"]

    sister = gMom.clone()
    brother = gDad.clone()
    sister.resetStats()
    brother.resetStats()
    crossover_nodes = np.unique(gMom['to'])
    for i in crossover_nodes:
      if Util.randomFlipCoin(Consts.CDefG1DListCrossUniformProb):
        sistemp = sister.getnode(i)
        brotemp = brother.getnode(i)
        sister.setnode(brotemp,i)
        brother.setnode(sistemp,i)

    return (sister, brother)
