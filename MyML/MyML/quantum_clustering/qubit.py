import numpy as np
from bitstring import BitArray

class qubit:

	def __init__(self,a=None,b=None,val=None):
		self.alpha=a
		self.beta=b
		self.value=val

	def collapse(self):
		# make random (i.e. "quantum physical") observation for each qubit
		observation = np.random.rand()

		# beta is chosen for collapse
		# if beta is bigger than observation, than state collapses to 1, otherwise to 0
		if self.beta >= observation:
			self.value = 1
		else:
			self.value = 0


	def rotation_gate(self,angle):
		qb=np.matrix([[self.alpha],[self.beta]])

		rot=np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

		qb=rot*qb

		self.alpha=qb.A1[0]
		self.beta=qb.A1[1]

	def angle_dist(self,otherQubitValue):
		return otherQubitValue * (np.pi / 2) - np.arctan(self.alpha / self.beta)

	def varAngleDist(self,otherQubit):
		if type(otherQubit) is not type(qubit()):
			raise Exception("otherQubitValue not of qubit type.")

		if otherQubit.value == self.value:
			return

		angle = self.angle_dist(otherQubit.value)
		angle=angle*(1/np.random.uniform(2,6))
		self.rotation_gate(angle)
		



class qubitString:

	def __init__(self,m):

		# size of qubit string
		self.size=m

		# floats can only be 32 or 64 bits
		self.canHaveFloat = True if self.size==32 or self.size==64 else False

		# create BitArray to hold value of qubit string
		self.arrayVal=BitArray(length=m)

		# initialization value for qubits
		initVal=1/np.sqrt(2)

		quString=list()
		for i in range(0,self.size):
			quString.append(qubit(initVal,initVal))

		self.quString=quString
		self.decVal=None
		self.binVal=None
		self.floatVal=None

	def collapse(self):
		self.bitString=list()
		for i in range(0,self.size):
			# observe and collapse qubit to a state
			self.quString[i].collapse()

			# save collapsed value to bit array
			if self.quString[i].value == 1:
				self.arrayVal.overwrite('0b1',i)
			else:
				self.arrayVal.overwrite('0b0',i)

		# update decimal value
		self.decVal=self.arrayVal.int
		self.binVal=self.arrayVal.bin
		if self.canHaveFloat : self.floatVal = self.arrayVal.float

	
	def rotation_gate(self,angle):
		for i,j in enumerate(quString):
			quString[i].rotation_gate(angle)


	def updateDec(self,num):
		# update array value
		self.arrayVal.int=num

		# update binary value
		self.binVal=self.arrayVal.bin

		# update qubits values
		for i in range(0,self.size):
			self.quString[i].value=int(self.binVal[i])
