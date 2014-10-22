#!/usr/bin/env python

import G1DConnections
import numpy as np
import random

if __name__=='__main__':

    pa=G1DConnections.G1DConnections(17)
    ma=G1DConnections.G1DConnections(17)


    layers=[2,4,1]
    ndLayers=np.array(layers)
    bias=[1,1,0]
    ndBias=np.array(bias)
    genomeList1=[]
    genomeList2=[]

#    pa.append([0,3,0.1])
#    pa.append([0,4,0.2])

#    ma.append([0,3,-0.1])
#    ma.append([0,4,-0.2])

    # ch1 = pa+ma
    # print "pa+ma", ch1.genomeList
    
    # print "pa", pa.genomeList
    # print "ma", ma.genomeList

    # ch2 = pa*10
    # print "pa*10", ch2.genomeList

    for ly in range(len(layers)-1):
        for toNeurode in range(layers[ly+1]):
            toNeurode+=int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
            print ndBias[0:ly+1].sum()
            for fromNeurode in range(layers[ly]+1):
                fromNeurode+=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
                genomeList1.append((fromNeurode, toNeurode, random.uniform(-10,10)))
                num_cumul+=1
    pa.genomeList=np.array(genomeList1, dtype=[('from','i'),('to','i'),('weight','f')])

    num_cumul=0
    for ly in range(len(layers)-1):
        for toNeurode in range(layers[ly+1]):
            toNeurode+=int(ndLayers[0:ly+1].sum()+ndBias[0:ly+1].sum())
            print ndBias[0:ly+1].sum()
            for fromNeurode in range(layers[ly]+1):
                fromNeurode+=int(ndLayers[0:ly].sum()+ndBias[0:ly].sum())
                genomeList2.append((fromNeurode, toNeurode, float(num_cumul)))
                num_cumul+=1    
    ma.genomeList=np.array(genomeList2, dtype=[('from','i'),('to','i'),('weight','f')])

    # genomeArray = np.array(genomeList, dtype=[('from','i'),('to','i'),('weight','f')])
    # toArray = genomeArray['to']
    # toSelected=toArray[np.equal(toArray,4)]
    # toIndex=np.nonzero(np.equal(toArray,4))
    # node=genomeArray[toIndex]

    nodeFromPa = pa.getnode(4)
    nodeFromMa = ma.getnode(4)

    pa.setnode(nodeFromMa,4)
    ma.setnode(nodeFromPa,4)
    
