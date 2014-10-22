import GAnnInitializators
import GAnnMutators
import GAnnCrossovers

# - G1DConnections defaults
CDefG1DConnMutIntMU = 2
CDefG1DConnMutIntSIGMA = 10

CDefG1DConnMutRealMU = 0
CDefG1DConnMutRealSIGMA = 1

CDefG1DConnMutator   = GAnnMutators.G1DConnMutateNodes
CDefG1DConnCrossover = GAnnCrossovers.G1DConnCrossoverNodes
CDefG1DConnInit      = GAnnInitializators.G1DConnInitializatorUniform
CDefG1DConnCrossUniformProb = 0.5
CDefMSETolerance = 1e-5
