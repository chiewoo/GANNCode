from random import uniform as rand_uniform, gauss as rand_gauss
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

    if genome.genomeList == []:
        totalConnections = int(np.sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)]))
        
        genomeList1=[]
    
        for ly in range(len(layers)-1):
            for toNeurode in range(layers[ly+1]):
                toNeurode += int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
                for fromNeurode in range(layers[ly]+1):
                    fromNeurode +=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
                    genomeList1.append((fromNeurode, toNeurode, rand_uniform(range_min, range_max)))
    
        genome.genomeList = np.array(genomeList1, dtype=[('from','i'),('to','i'),('weight','f')])
    else:
        totalConnections = len(genome.genomeList)
        if totalConnections != genome.genomeSize:
            print "%% Warning: genome.genomeSize and the length of genome.genomeList mismatch!"
        for conn in genome.genomeList:
            conn[2] = rand_uniform(range_min, range_max)
    
def G1DConnInitializatorGaussian(genome, **args):
    """ Real initialization function of G1DConnections
    
    This initializator accepts the *rangemin* and *rangemax* genome parameters.
    
    """
    range_min = genome.getParam("rangemin", -50)
    range_max = genome.getParam("rangemax", 50)
    mu = genome.getParam("gauss_mu", 0)
    sigma = genome.getParam("gauss_sigma", 1)
    #    print "mu=", mu, "sigma=", sigma

    layers = genome.getParam("layers", [2,4,1])
    bias = genome.getParam("bias", [1,1,0])

    ndLayers=np.array(layers)
    ndBias=np.array(bias)

    if genome.genomeList == []:
        totalConnections = int(np.sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)]))
        
        genomeList1=[]
    
        for ly in range(len(layers)-1):
            for toNeurode in range(layers[ly+1]):
                toNeurode += int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
                for fromNeurode in range(layers[ly]+1):
                    fromNeurode +=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
                    weight = min(range_max, rand_gauss(mu, sigma))
                    weight = max(range_min, weight)
                    genomeList1.append((fromNeurode, toNeurode, weight))
    
        genome.genomeList = np.array(genomeList1, dtype=[('from','i'),('to','i'),('weight','f')])
    else:
        totalConnections = len(genome.genomeList)
        if totalConnections != genome.genomeSize:
            print "%% Warning: genome.genomeSize and the length of genome.genomeList mismatch!"
        for conn in genome.genomeList:
            weight = min(range_max, rand_gauss(mu, sigma))
            weight = max(range_min, weight)
            conn[2] = weight

            
