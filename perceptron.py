"""
Author: Grant Holmes
Date: 09/18/2020
Description: Single layer perceptron neural network.
"""

import numpy as np

class Perceptron:
	"""
	Single layer perceptron neural network.
	"""
	def __init__(self, input_size):
		"""
		input_size is the number of inputs to the perceptron.
		"""
		np.random.seed(42)
		self.training_cycles = 0
		self.synapses = np.random.rand(input_size, 1)
		self.bias = np.random.rand(1)
		self.lr = 0

	def sigmoid(self, x):
		"""
		Sigmoid logistic function.
		"""
		return 1/(1+np.exp(-x))

	def sigmoid_der(self, x):
		"""
		Sigmoid logistic function derivative.
		"""
		return self.sigmoid(x)*(1-self.sigmoid(x))

	def think(self, inputs):
		"""
		Takes in an input and runs it through perceptron, outputting values based on prior training.
		"""
		inputs = inputs.astype(np.float64)
		output = self.sigmoid(np.dot(inputs, self.synapses) + self.bias) 
		return output

	def cost(self, output, training_outputs):
		"""
		Cost function used for gradient descent.
		"""
		return output - training_outputs

	def train(self, training_inputs, training_outputs, training_cycles, learning_rate=0.05):
		"""
		Train perceptron using input data. Training cycles if number of time to iterate. Learning rate
			determines how much the perceptron can learn from a single training episdoe.
		"""
		self.lr = learning_rate
		self.training_cycles = training_cycles		

		for training_cycle in range(training_cycles):
			output = self.think(training_inputs)

			error = self.cost(output, training_outputs)
			# print(error.sum())

			dcost_dpred = error
			dpred_doutput = self.sigmoid_der(output)

			output_delta = dcost_dpred*dpred_doutput
			inputs = training_inputs.T
			self.synapses -= self.lr*np.dot(inputs, output_delta)

			for num in output_delta:
				self.bias -= self.lr * num

