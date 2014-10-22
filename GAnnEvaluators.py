from pyevolve import Util
from pyfann import libfann
import GAnnConsts

def evaluateMSE(genome, **args):
	# FANN library is used for Artificial Neural Network structrue.
	ann=args['network']
	data=args['data']
	ann.set_weight_array(genome.toList())
	MSE=ann.test_data(data)

	return float(1)-(MSE)#+GAnnConsts.CDefMSETolerance)

def evaluateMMSE(genome, **args):
	# FANN library is used for Artificial Neural Network structrue.
	ann=args['network']
	data=args['data']
	ann.set_weight_array(genome.toList())
	#MSE=ann.test_data(data)
	data_inputs=data.get_input()
	data_outputs=data.get_output()
	g_factor=0.5
	c_factor=0.5
	g_base=0.1
	c_base=0.1
	g_penalty=g_factor*float(data.length_train_data())/float(data_outputs.count([1]))
	c_penalty=c_factor*float(data.length_train_data())/float(data_outputs.count([0]))
	ann.reset_MSE()
	add_MSE=0.0
	for i in xrange(data.length_train_data()):
		ann.test(data_inputs[i],data_outputs[i])
		outputs=ann.run(data_inputs[i])
		for j in xrange(data.num_output_train_data()):
			diff=outputs[j]-data_outputs[i][j]
			diff*=diff
			if data_outputs[i][j]==0.0 and diff>c_base*c_base:
				add_MSE+=c_penalty*(diff-c_base*c_base)/(1-c_base*c_base)
			elif data_outputs[i][j]==1.0 and diff>g_base*g_base:
				add_MSE+=g_penalty*(diff-g_base*g_base)/(1-g_base*g_base)
	MSE=(ann.get_MSE()+(add_MSE/float(data.length_train_data())))/(g_factor+c_factor+1.0)

	return float(1)/(MSE)#+GAnnConsts.CDefMSETolerance)


