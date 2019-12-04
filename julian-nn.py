#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Julian Allchin
"""

import numpy as np
import matplotlib.pyplot as plt

"""Training set data and testing set data"""
training_set="./mnist_train.csv" #short training set
test_set="./mnist_test_10.csv"       # short test set

def sigmoid(x):
	return (1/(1+np.exp(-x)))
	
def derviativeSigmoid(x):
	return sigmoid(x) * (1 + sigmoid(x))

"""Artificial Neural Network Class"""
class neuralNetwork:
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		
		# Building layers list
		# [inputNodes, nodesInHiddenLayer1, nodesInHiddenLayer2,..., outputNodes]
		self.layers = [inputNodes]

		for x in hiddenNodes:
			self.layers.append(x)
		self.layers.append(outputNodes)

		self.error = []
		self.weights = []
		self.layerOutput = []
		self.cost = [0] * (len(self.layers) - 1)
		self.lr = learningRate
		
		for i in range(1, len(hiddenNodes) + 2):
			self.weights.append(np.random.normal(0.0, pow(self.layers[i-1], -0.5), (self.layers[i], self.layers[i-1])))
			print("Creating weight matrix of size " + str(self.weights[len(self.weights) -1].shape))

		pass
	 
	def forwardProp(self, X):

		# For each layer
		for i in range(0, len(self.layers)-1):
			# If its the first one
			if i == 0:
				# Forward prop input by weights
				self.layerOutput.append(sigmoid(np.dot(X, self.weights[i].T)))

			# Not the first one	
			else:

				# Propagate through rest
				self.layerOutput.append(sigmoid(np.dot(self.layerOutput[i-1], self.weights[i].T)))	
			
		# Set y_hat to the last (output) nodes
		self.y_hat = self.layerOutput[len(self.layerOutput) - 1 ]

		pass
		
	def backProp(self, X, y):

		# application of the chain rule to find derivative of the loss 
		#  function with respect to weights2 and weights1
		
		m = self.y_hat.shape[1]
		cost = -1 / m * (np.dot(y, np.log(self.y_hat).T) + np.dot(1 - y, np.log(1 - self.y_hat).T))
		self.error.append(np.squeeze(cost))
		
		baseCost = (self.y_hat - y) * derviativeSigmoid(self.y_hat)
		
		# For each layer
		for i in range(len(self.weights)-1, 0, -1):
			# Generate a cost from the baseCost
			if i == len(self.weights)-1:
				print(i, "baseCost")
				self.cost[i] = baseCost
			else:
				print(i, "adding to")
				
				# Current cost = previous cost * W * R'(y_hati)
				self.cost[i] = self.cost[i+1] * self.weights[i] * derviativeSigmoid(self.layerOutput[i])
			
				# Dotting the next layer
				if i == 0:
					self.cost[i] = self.cost[i] * X
				else:
					print(i, "nextLayer")

					self.cost[i] = self.cost[i] * self.layerOutput[i-1]

			# Mutliplying by the learning rate
			self.cost[i] = self.cost[i] * self.lr

			print(self.weights[i].shape, self.cost[i].T.shape)

			# Nudging it
			self.weights[i] += self.cost[i].T


		# tempCalc = 2*(y - self.y_hat) * derviativeSigmoid(self.y_hat)
		
		# d_who = self.lr * np.dot(self.layer1.T, tempCalc)
		# d_wih = self.lr * np.dot(X.T, (np.dot(tempCalc, self.who) * derviativeSigmoid(self.layer1)))
		
		# self.wih += d_wih.T
		# self.who += d_who.T
				
	# train the neural network
	def train(self, X, y):
		self.forwardProp(X)
		self.backProp(X, y)
			   
	def guess(self, X):
		 # calculate signals into hidden layer: (200 x 784) * (784 x 1) => (200 x 1)
		hiddenOut = sigmoid(np.dot(self.wih, X))
		# calculate signals into final output layer (10 x 200) * (200 x 1) => (10 x 1)
		finalOut = sigmoid(np.dot(self.who, hiddenOut))
		return finalOut
	
 
	
"""Create instance of neural network """   
inputNodeCount = 784
hiddenNodesCount = [200, 300]
outputNodeCount = 10
learningRate = 0.001
nn = neuralNetwork(inputNodeCount,hiddenNodesCount,outputNodeCount, learningRate)


"""Train the ANN"""
# load the mnist training data CSV file into a list
training_data_file = open(training_set, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 5
for e in range(epochs):
	for record in training_data_list:      
		all_values = record.split(',') # split the record by the ',' commas
					
		inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #scale
		# create the target output values (all 0.01, except ans = 0.99)
		targets = np.zeros(outputNodeCount) + 0.01
		targets[int(all_values[0])] = 0.99 # all_values[0] answer for input
		X=np.array(inputs, ndmin=2) # create a 784 x 1 vector from input
		nn.train(X, targets)

		plt.clf()

		# axes = plt.gca()
		# axes.set_ylim([0,1])
		plt.plot(nn.error)
		plt.draw()
		plt.pause(.001)


# """Test ANN on new samples"""
# test_data_file = open(test_set, 'r') # load the mnist test data CSV file into a list
# test_data_list = test_data_file.readlines()
# test_data_file.close()
# scorecard = []
# # go through all the records in the test data set
# for record in test_data_list: 
# 	all_values = record.split(',') # split the record by the ',' commas
# 	correct_label = int(all_values[0])                        
# 	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs
# 	outputs = nn.guess(inputs) # ANN's guess as to the number
# 	label = np.argmax(outputs) # the index of the highest value corresponds to the answer
# 	if (label == correct_label):
# 		# network's answer matches correct answer, add 1 to scorecard
# 		print(correct_label, "correct")
# 		scorecard.append(1)
# 	else:
# 		# network's answer doesn't match correct answer, add 0 to scorecard
# 		print(correct_label, "wrong")
# 		scorecard.append(0)
# 		pass
# 	pass

# """Calculate the performance score, the fraction of correct answers"""
# print (scorecard)
# scorecard_array = np.asarray(scorecard)
# print ("performance =", scorecard_array.sum() * 100./ scorecard_array.size, "%")