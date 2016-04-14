import numpy as np
from qubit import * 

class Oracle:

	def __init__(self):
		self.qstrings=list()
		self.score=np.inf

	def setScore(self,score):
		self.score=score

	def initialization(self,n,m):
		self.stringNum=n
		self.stringSize=m

		# create n qubit strings with m qubits each
		for i in range(0,n):
			self.qstrings.append(qubitString(m))

	def collapse(self):
		# make observation and collapse the state of all qubits of all strings
		for i in range(0,self.stringNum):
			self.qstrings[i].collapse()

	def getFloatArrays(self):
		# returns 1-dim numpy.array with all string values in row as float
		res=list()
		for i,j in enumerate(self.qstrings):
			res.append(j.floatVal)

		return np.array(res)

	def getIntArrays(self):
		# returns 1-dim numpy.array with all string values in row as integer
		res=list()
		for i,j in enumerate(self.qstrings):
			res.append(j.decVal)

		return np.array(res)

	def setIntArrays(self,intArray):
		# receives 1-dim numpy.array with all string values in row
		if type(intArray) is not type(np.ndarray([])):
			raise Exception("Input not of type numpy.array.")

		intArray=np.around(intArray).astype(int)	
		for i,j in enumerate(self.qstrings):
			j.updateDec(intArray[i])


	def QuantumGateStep(self,otherOracle):
		if type(otherOracle) != type(Oracle()):
			raise Exception("otherOracle not of Oracle type.")

		# for each qubit string
		for qs in range(0,self.stringNum):
			# for each qubit
			for qb in range(0,self.stringSize):
				# check if self qubit's value is equal to other qubit's value
				if self.qstrings[qs].quString[qb].value == otherOracle.qstrings[qs].quString[qb].value:
					continue
				else: # if it's not, apply the Variable Angle Distance Quantum Rotation
					self.qstrings[qs].quString[qb].varAngleDist(otherOracle.qstrings[qs].quString[qb])