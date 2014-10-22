from random import random as rand_random
from GAnnCrossovers import G1DConnCrossoverWeights, G1DConnCrossoverNodes
#from pyevolve import GenomeBase,G1DList
import G1DConnections
import numpy as np

n_in=5
n_out=1
n_hid=1
n_hneu=[3]
n_neu=[n_in]+n_hneu+[n_out]
temp=0
sum=0
to_list=[]
momgenome=G1DConnections.G1DConnections()
dadgenome=G1DConnections.G1DConnections()
for i in xrange(1,len(n_neu)):
  sum=sum+n_neu[i-1]+1
  for toNode in xrange(sum,sum+n_neu[i]):
    to_list=to_list+[toNode]
    for fromNode in xrange(temp,sum-1):
      momgenome.append((fromNode,toNode,rand_random()))
      dadgenome.append((fromNode,toNode,rand_random()))
  temp=sum
momgenome.genomeList=np.array(momgenome.genomeList, dtype=[('from','i'),('to','i'),('weight','f')])
dadgenome.genomeList=np.array(dadgenome.genomeList, dtype=[('from','i'),('to','i'),('weight','f')])
print "######  Network setting  ######"
print "Number of input: ",n_in
print "Number of output: ",n_out
print "Number of hidden layer: ",n_hid
print "Number of neurons for each hidden layer: ",n_hneu
print "###############################"
print "toNode list: ",to_list
print "###############################"
print "         Parents:"
print "###############################"
print "  MomGenome:\t\t  DadGenome:"
for i in xrange(len(momgenome.genomeList)):
  (x,y,z)=momgenome.genomeList[i]
  (u,v,w)=dadgenome.genomeList[i]
  print "(%2d, %2d, %1.2f)\t\t(%2d, %2d, %1.2f)" % (x,y,z,u,v,w)
print ""
print "###############################"
print "G1DConnCrossoverWeights() test:"
print "###############################"
(sister,brother)=G1DConnCrossoverWeights(momgenome,Mom=momgenome,Dad=dadgenome)
print "    Sister:\t\t    Brother:"
for i in xrange(len(sister.genomeList)):
  (x,y,z)=sister.genomeList[i]
  (u,v,w)=brother.genomeList[i]
  if(z==momgenome.genomeList[i][2]):
    sis="Mom"
    bro="Dad"
  else:
    sis="Dad"
    bro="Mom"
  print "%s (%2d, %2d, %1.2f)\t%s (%2d, %2d, %1.2f)" % (sis,x,y,z,bro,u,v,w)
print ""
print "###############################"
print "G1DConnCrossoverNodes() test:"
print "###############################"
(sister,brother)=G1DConnCrossoverNodes(momgenome,Mom=momgenome,Dad=dadgenome,to_list=to_list)
print "    Sister:\t\t    Brother:"
for i in xrange(len(sister.genomeList)):
  (x,y,z)=sister.genomeList[i]
  (u,v,w)=brother.genomeList[i]
  if(z==momgenome.genomeList[i][2]):
    sis="Mom"
    bro="Dad"
  else:
    sis="Dad"
    bro="Mom"
  print "%s (%2d, %2d, %1.2f)\t%s (%2d, %2d, %1.2f)" % (sis,x,y,z,bro,u,v,w)

