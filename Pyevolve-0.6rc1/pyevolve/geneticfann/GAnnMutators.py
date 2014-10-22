import numpy
from random import randint as rand_randint, gauss as rand_gauss, uniform as rand_uniform
from random import choice as rand_choice
from random import sample as rand_sample
from pyevolve import Consts
from pyevolve import Util
from pyfann import libfann
import GAnnConsts

#def MutationCriterion(criterion)
#	if criterion == 'gaussian': 
#	return criterion

def G1DConnUnbiasedMutateWeights(genome, **args):
	## args['mutation_conns'] takes the list of indices of connections to be mutated 
	if not args['mutation_conns']:
		mutation_conn_indices=rand_sample(numpy.arange(len(genome)),int(args['pmut']*len(genome)))
		#		print "Indices of mutation weights:"
		#		print mutation_conn_indices
	else:
		mutation_conn_indices = args['mutation_conns']
		#		print "Indices of mutation weights:"
		#		print mutation_conn_indices

	new_mutation_list=[]
	for it in mutation_conn_indices:
		new_mutation_list.append(rand_uniform(genome.getParam("rangemin", Consts.CDefRangeMin),genome.getParam("rangemax", Consts.CDefRangeMax)))

	numpy.put(genome['weight'],mutation_conn_indices,new_mutation_list)

	return genome

def G1DConnBiasedMutateWeights(genome, **args):
	## args['mutation_conns'] takes the list of indices of connections to be mutated 
	if not args['mutation_conns']:
		mutation_conn_indices=rand_sample(numpy.array(len(genome)-1),int(args['pmut']*len(genome)))
		#		print "Indices of mutation weights:"
		#		print mutation_conn_indices
	else:
		mutation_conn_indices = args['mutation_conns']
		#		print "Indices of mutation weights:"
		#		print mutation_conn_indices

	mu = genome.getParam("gauss_mu")
	sigma = genome.getParam("gauss_sigma")

	new_mutation_list=[]
	for it in mutation_conn_indices:
		final_value = genome['weight'][it] + rand_gauss(mu, sigma)

		#final_value = min(final_value, genome.getParam("rangemax", Consts.CdefRangeMax))
		#final_value = max(final_value, genome.getParam("rangemin", Consts.CdefRangeMin))

		new_mutation_list.append(final_value)
	
	numpy.put(genome['weight'],[mutation_conn_indices],[new_mutation_list])
		
	return genome

def G1DConnMutateNodes(genome, **args):
	## args['mutation_nodes'] takes the list of node numbers to be mutated
	if not args['mutation_nodes']:
		mutations=int(args['pmut']*len(numpy.unique(genome['to'])))
		mutation_node_numbers = rand_sample(numpy.unique(genome['to']), mutations)
		#		print "randomly chosen mutation nodes:"
		#		print mutation_node_numbers
		
	else:
		mutation_node_numbers = list(args['mutation_nodes'])
		#		print "selected mutation nodes:"
		#		print mutation_node_numbers
		mutations = len(arg['mutation_nodes'])

	for it in mutation_node_numbers:
		mutation_nodes = genome.getnode(it)
		#		print "connections of mutation nodes"
		#		print mutation_nodes
		mutation_conn_indices=getMutateConnIndex(genome, mutation_nodes)
		G1DConnBiasedMutateWeights(genome, mutation_conns=mutation_conn_indices)	

	return int(mutations)

def getMutateConnIndex(genome, connections):
	# returns list of indecies of given connections in genome. 
	conn_index = []
	for conn in connections:
		toIndex = numpy.nonzero(numpy.equal(genome['to'],conn['to']))
		toIndex = toIndex[0]
		#print 'test'
		#print toIndex
		#print genome[toIndex]
		#print genome['from'][toIndex]
		index = numpy.searchsorted(genome['from'][toIndex],conn['from'])
		#print "index:%i" % index
		#print toIndex[index]
		conn_index.append(toIndex[index])

	return conn_index
		
def getMutateNodeIndex(num_neurons,layer_num,node_num):
	# num_neurons is a list which contains number of neurons at each layer.',' separation is used.
	# last neuron in each layer is bias neuron
	num_hidden_neurons = num_neurons[1:-1]
	#for neurons in num_neurons:
	#node_index = 
	return node_index

def evaluateMSE(genome, **args):
	# FANN library is used for Artificial Neural Network structrue.
	ann=args['network']
	data=args['data']
	ann.set_weight_array(genome.toList())
	MSE=ann.test_data(data)

	return float(1)/(MSE)#+GAnnConsts.CDefMSETolerance)


