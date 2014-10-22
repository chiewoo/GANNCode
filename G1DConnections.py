from pyevolve.GenomeBase import GenomeBase, G1DBase
from pyevolve import Consts
import GAnnConsts
import copy
import numpy as np
class G1DConnections(GenomeBase, G1DBase):
    def __init__(self, size=10, cloning=False):
        """ The initializator of G1DConnections representation,
        size parameter must be specified """
        GenomeBase.__init__(self)
        G1DBase.__init__(self, size)
        self.genomeList=[]
#        self.genomeList=np.empty((3,size),dtype=[('from','i'),('to','i'),('weight','f')])
        if not cloning:
            self.initializator.set(GAnnConsts.CDefG1DConnInit)
            self.mutator.set(GAnnConsts.CDefG1DConnMutator)
            self.crossover.set(GAnnConsts.CDefG1DConnCrossover)

    def __mul__(self, other):
        """ Multiply every element of G1DConnections by "other" """
        newObj = self.clone()
        if isinstance(other, G1DConnections):
            for i in xrange(len(newObj)):
                newObj[i][2] *= (other[i])[2]
        else:
            for i in xrange(len(newObj)):
                newObj[i][2] *= other
        return newObj

    def __add__(self, other):
        """ Plus every element of G1DConnections by "other" """
        newObj = self.clone()
        if isinstance(other, G1DConnections):
            for i in xrange(len(newObj)):
                (newObj[i])[2] += (other[i])[2]
        else:
            for i in xrange(len(newObj)):
                (newObj[i])[2] += other
        return newObj

    def __sub__(self, other):
        """ Plus every element of G1DConnections by "other" """
        newObj = self.clone()
        if isinstance(other, G1DConnections):
            for i in xrange(len(newObj)):
                (newObj[i])[2] -= (other[i])[2]
        else:
            for i in xrange(len(newObj)):
                (newObj[i])[2] -= other
        return newObj

    def __repr__(self):
        """ Return a string representation of Genome """
        ret = GenomeBase.__repr__(self)
        ret += "- G1DConnections\n"
        ret += "\tList size:\t %s\n" % (self.getListSize(),)
        ret += "\tList:\t\t %s\n\n" % (self.genomeList,)
        return ret

    def copy(self, g):
        """ Copy genome to 'g'
        
      Example:
      >>> genome_origin.copy(genome_destination)
      
      :param g: the destination G1DConnections instance

      """
        GenomeBase.copy(self, g)
        #G1DBase.copy(self, g)
        g.genomeSize=self.genomeSize
        g.genomeList=copy.deepcopy(self.genomeList)
   
    def clone(self):
        """ Return a new instace copy of the genome
        
      :rtype: the G1DConnections clone instance
      
      """
        newcopy = G1DConnections(self.genomeSize, True)
        self.copy(newcopy)
        return newcopy
    
    def getnode(self, toNodeId):
        """ Return incoming connections to toNodeID """
        #   genomeArray = np.array(genomeList, dtype=[('from','i'),('to','i'),('weight','f')])
        toArray = self.genomeList['to']
        #   toSelected=toArray[np.equal(toArray,toNodeId)]
        toIndex=np.nonzero(np.equal(toArray,toNodeId))
        node=self.genomeList[toIndex]
        return node
    
    def setnode(self, node, toNodeId):
        """ Set incoming connections to toNodeId with node"""
        toArray = self.genomeList['to']
        toIndex=np.nonzero(np.equal(toArray,toNodeId))        
        self.genomeList[toIndex]=node

    def toList(self):
        """ Convert genomeList to a list of tuples """
        return self.genomeList.tolist()


if __name__=='__main__':
    import random

    # neural network topology
    layers=[2,4,1]
    ndLayers=np.array(layers)
    bias=[1,1,0]
    ndBias=np.array(bias)
    
    totalConnections = int(np.sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)]))
    # set variables
    genomeList1=[]
    genomeList2=[]

    # instantiate G1DConnections for ma and pa
    pa=G1DConnections(totalConnections)
    ma=G1DConnections(totalConnections)

    # generate network connections of FANN for testing
    for ly in range(len(layers)-1):
        for toNeurode in range(layers[ly+1]):
            toNeurode+=int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
            for fromNeurode in range(layers[ly]+1):
                fromNeurode+=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
                genomeList1.append((fromNeurode, toNeurode, random.uniform(-10,10)))
    # convert the connections to numpy object and assign it into genomList of the G1DConnection object
    pa.genomeList=np.array(genomeList1, dtype=[('from','i'),('to','i'),('weight','f')])
    print "##########################"
    print "# genome of PAPA         #"
    print "##########################"
    print pa.genomeList


    num_cumul=0
    for ly in range(len(layers)-1):
        for toNeurode in range(layers[ly+1]):
            toNeurode+=int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
            for fromNeurode in range(layers[ly]+1):
                fromNeurode+=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
                genomeList2.append((fromNeurode, toNeurode, float(num_cumul)))
                num_cumul+=1    
    # convert the connections to numpy object and assign it into genomList of the G1DConnection object
    ma.genomeList=np.array(genomeList2, dtype=[('from','i'),('to','i'),('weight','f')])
    print "##########################"
    print "# genome of MAMA         #"
    print "##########################"
    print ma.genomeList
    # get node=4 
    nodeFromPa = pa.getnode(4)
    nodeFromMa = ma.getnode(4)

    # exchange
    print "###############################################"
    print "# example of G1DConnections.getnode()         #"
    print "# example of G1DConnections.setnode()         #"
    print "# ==> exchaning connections for nodeID 4      #"
    print "###############################################"

    pa.setnode(nodeFromMa,4)
    ma.setnode(nodeFromPa,4)

    print "##########################"
    print "# after exchanging       #"
    print "# genome of PAPA         #"
    print "##########################"
    print pa.genomeList
    print

    print "##########################"
    print "# after exchanging       #"
    print "# genome of MAMA         #"
    print "##########################"
    print ma.genomeList
    print
