from random import uniform as rand_uniform
import numpy as np

def G1DConnInitializatorUniform(genome, **args):
    """ Real initialization function of G1DConnections
    
    This initializator accepts the *rangemin* and *rangemax* genome parameters.
    
    """
    range_min = genome.getParam("rangemin", -50)
    range_max = genome.getParam("rangemax", 50)
    layers = genome.getParam("layers", [2,4,1])
    bias = genome.getParam("bias", [1,1,0])

    ndLayers=np.array(layers)
    ndBias=np.array(bias)
    
    totalConnections = int(np.sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)]))
    
#    if (totalConnections != genome.genomeSize):
#        print "genomeSize and the number of connectons mismatched"
#        return

    genomeList1=[]

    for ly in range(len(layers)-1):
        for toNeurode in range(layers[ly+1]):
            toNeurode += int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
            for fromNeurode in range(layers[ly]+1):
                fromNeurode +=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
                genomeList1.append((fromNeurode, toNeurode, rand_uniform(range_min, range_max)))

    genome.genomeList = np.array(genomeList1, dtype=[('from','i'),('to','i'),('weight','f')])
                
    #   for ly in range(len(layers)-1):
    #       for toNeurode in range(layers[ly+1]):
    #         toNeurode+=int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
    #         for fromNeurode in range(layers[ly]+1):
    #             fromNeurode+=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
    #             genomeList1.append((fromNeurode, toNeurode, rand_uniform(range_min, range_max))
