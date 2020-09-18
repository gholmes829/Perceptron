"""
Author: Grant Holmes
Date: 09/18/2020
Description: Example of single layer perceptron neural network usage.
"""

import numpy as np
from time import time
from itertools import product
from perceptron import Perceptron

def main():

	training_inputs = np.array([[0,1,1,0],
								[1,1,1,1],
								[1,0,0,1],
		                        [0,1,1,1],
								[0,0,0,1],
								[1,0,1,0]])

	training_outputs = np.array([[0,1,1,1,1,0]]).T

	num_entries = len(training_inputs)
	entry_size = len(training_inputs[0])
	
	combos = list(product([0, 1], repeat=entry_size))

	learning_rate = 0.05
	training_cycles = 20000

	perceptron = Perceptron(entry_size)

	print("o" + "="*38 + "o")
	print("| EXAMPLE PERCEPTRON NEURAL NETOWRK!!! |")
	print("o" + "="*38 + "o")
	print("\nTraining entries (n=" + str(num_entries) + "):")
	for i in range(num_entries):
		print("\t" + str(i+1) + ") " + str(training_inputs[i]) + ": " + str(training_outputs[i][0]))
	print("\nInitial Weights: ")
	for i in range(entry_size):
		print("\t" + str(i+1) + ") " + str(round(perceptron.synapses[i][0], 5)))
	print("\nInitial Bias = " + str(round(perceptron.bias[0], 5)))
	print("\nTraining for " + str(training_cycles) + " cycles...")

	timer = time()
	perceptron.train(training_inputs, training_outputs, training_cycles, learning_rate=learning_rate)
	time_elapsed = round(time()-timer, 3)
	
	print("Training complete after " + str(time_elapsed) + " secs...")
	print("\nFinal Weights After Training:")
	for i in range(entry_size):
		print("\t" + str(i+1) + ") " + str(round(perceptron.synapses[i][0], 5)))
	print("\nFinal Bias After Training = " + str(round(perceptron.bias[0], 5)))
	print()
	print("Final Results (n=" + str(len(combos)) + "):")
	c = 1
	for i in combos:	
		inputs = np.array(i)
		output = perceptron.think(inputs)
		rounded_output = round(output[0], 0)
		print("\t" + " "*(len(str(len(combos)))-len(str(c))) + str(c) + ") " + str(inputs) + ": " + str(int(round(output[0], 0))) + " " + "(" + str(round(output[0]*100, 2)) +"%)")
		c+=1
	print("\nPercentage values represent confidence of output for each possible binary entry.")
	print("\nExiting Successfully...")
	
if __name__ == "__main__":
	main()
	
