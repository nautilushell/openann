#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Julian Allchin
"""

import numpy as np
import matplotlib.pyplot as plt

"""Training set data and testing set data"""
training_set="./mnist_train_100.csv" #short training set
test_set="./mnist_test_10.csv"       # short test set

def sigmoid(x):
	return (1/(1+np.exp(-x)))
	
def derviativeSigmoid(x):
	return (x) * (1 + (x))

"""Artificial Neural Network Class"""
class NeuralNetwork:
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

	def accuracy(self, y):
		prob = np.copy(self.y_hat)
		prob[prob > 0.5] = 1
		prob[prob <= 0.5] = 0

		return (prob == y).all(axis=0).mean()

	def variance(self, y):
		return np.sum(np.square(y - self.y_hat))
	 
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

		# Use chain rule to find derivative of the loss with respect to weights
		
		self.error.append(self.variance(y))
		
		# For each layer   
		for i in range(len(self.layers)-2, 0, -1):
			# Generate a cost from the baseCost
			if i == len(self.layers)-2:
				errorsum = 2 * (y - self.y_hat) * derviativeSigmoid(self.y_hat)
				prevError=errorsum.T
			else:
				errorsum = (np.dot(self.weights[i+1].T,prevError) * derviativeSigmoid(self.layerOutput[i].T)).T
				prevError=errorsum

			# Dotting the next layer
			if i == 0:
				self.cost[i] = np.dot(X.T, errorsum).T
			else:
				self.cost[i] = np.dot(self.layerOutput[i-1].T, errorsum).T

			# Mutliplying by the learning rate
			self.cost[i] = self.cost[i] * self.lr

			# Nudging it
			self.weights[i] += self.cost[i]

				
	# train the neural network
	def train(self, X, y):
		self.forwardProp(X)
		self.backProp(X, y)

	# after training save weights
	def save(self):
		with open('weights.npy','wb') as f:
			np.save(f, self.weights)
		f.close()
		print("Saved weights")

	# after training save weights
	def load(self, file):
		with open(file, 'rb') as f:
			self.weights = np.load(f)
		f.close()
			   
	def guess(self, X):
		# For each layer
		for i in range(0, len(self.layers)-1):
			# If its the first one
			if i == 0:
				# Forward prop input by weights
				lastOutput = sigmoid(np.dot(X, self.weights[i].T))

			# Not the first one	
			else:

				# Propagate through rest
				lastOutput = sigmoid(np.dot(lastOutput, self.weights[i].T))
			
		# Set y_hat to the last (output) nodes
		return lastOutput