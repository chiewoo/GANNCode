import sys
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
f = open('CrossoverResults.txt','w')
f.write( "######  Network setting  ######"+'\n')
f.write( "Number of input: "+str(n_in)+'\n')
f.write( "Number of output: "+str(n_out)+'\n')
f.write( "Number of hidden layer: "+str(n_hid)+'\n')
f.write( "Number of neurons for each hidden layer: "+str(n_hneu)+'\n')
f.write( "###############################"+'\n')
f.write( "toNode list: "+str(to_list)+'\n')
f.write("###############################"+'\n')
f.write( "         Parents:"+'\n')
f.write("###############################"+'\n')
f.write("  MomGenome:\t\t  DadGenome:"+'\n')
for i in xrange(len(momgenome.genomeList)):
  (x,y,z)=momgenome.genomeList[i]
  (u,v,w)=dadgenome.genomeList[i]
  f.write( "(%2d, %2d, %1.2f)\t\t(%2d, %2d, %1.2f)" % (x,y,z,u,v,w)+'\n')
f.write(""+'\n')
f.write( "###############################"+'\n')
f.write("G1DConnCrossoverWeights() test:"+'\n')
f.write( "###############################"+'\n')
(sister,brother)=G1DConnCrossoverWeights(momgenome,Mom=momgenome,Dad=dadgenome)
f.write( "    Sister:\t\t    Brother:"+'\n')
for i in xrange(len(sister.genomeList)):
  (x,y,z)=sister.genomeList[i]
  (u,v,w)=brother.genomeList[i]
  if(z==momgenome.genomeList[i][2]):
    sis="Mom"
    bro="Dad"
  else:
    sis="Dad"
    bro="Mom"
  f.write("%s (%2d, %2d, %1.2f)\t%s (%2d, %2d, %1.2f)" % (sis,x,y,z,bro,u,v,w)+'\n')
f.write( ""+'\n')
f.write( "###############################"+'\n')
f.write( "G1DConnCrossoverNodes() test:"+'\n')
f.write( "###############################"+'\n')
(sister,brother)=G1DConnCrossoverNodes(momgenome,Mom=momgenome,Dad=dadgenome,to_list=to_list)
f.write( "    Sister:\t\t    Brother:"+'\n')
for i in xrange(len(sister.genomeList)):
  (x,y,z)=sister.genomeList[i]
  (u,v,w)=brother.genomeList[i]
  if(z==momgenome.genomeList[i][2]):
    sis="Mom"
    bro="Dad"
  else:
    sis="Dad"
    bro="Mom"
  f.write("%s (%2d, %2d, %1.2f)\t%s (%2d, %2d, %1.2f)" % (sis,x,y,z,bro,u,v,w)+'\n')
f.close()
