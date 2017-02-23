import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('action', help='imports - tests all imports', type=str, nargs='+')
args = parser.parse_args()

def eac_imports():
	import MyML.cluster.K_Means3 as myKM
	import MyML.utils.partition as mpart
	import MyML.metrics.accuracy as acc
	import MyML.cluster.linkage as linkage
	import MyML.EAC.eac_new as eac
	import MyML.EAC.rules as rules
	import MyML.utils.profiling as prof

	pass

def quantum_imports():
	import MyML.quantum_clustering.Horn
	import MyML.quantum_clustering.oracle
	import MyML.quantum_clustering.qubit
	import MyML.quantum_clustering.QK_Means
	import MyML.quantum_clustering.K_Means_wrapper
	import MyML.quantum_clustering.gauss_mixture

def sl_gpu_imports():
	import MyML.cudasl.build
	import MyML.cudasl.connected_components
	import MyML.cudasl.mst

def all_imports():
	print 'Testing EAC imports'
	eac_imports()

	print 'Testing quantum clustering imports'
	quantum_imports()

	print 'Testing GPU SL imports'
	sl_gpu_imports()

actions = {'eac_imports': eac_imports,
		   'quantum_imports': quantum_imports,
		   'sl_gpu_imports': sl_gpu_imports,
		   'all_imports': all_imports}

if __name__ == '__main__':
	for action in args.action:
		print 'Test action: {}'.format(action)
		actions[action]()
	print 'Exiting...'
	sys.exit(0)