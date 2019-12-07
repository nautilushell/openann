#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Julian Allchin
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
	return (1/(1+np.exp(-x)))

def derviativeSigmoid(x):
	return (x * (1 - x))

"""Artificial Neural Network Class"""
class NeuralNetwork:
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

		# Building layers list
		# [inputNodes, nodesInHiddenLayer1, nodesInHiddenLayer2,..., outputNodes]
		self.layerNodeCount = [inputNodes]

		for x in hiddenNodes:
			self.layerNodeCount.append(x)
		self.layerNodeCount.append(outputNodes)

		self.lastvariance = 100
		self.error = []
		self.weights = []
		self.lr = learningRate

		print("Input layer size: "+ str(self.layerNodeCount[0]))
		# Create weight matrices between each neural layer level
		for i in range(0, len(hiddenNodes) + 1):
			self.weights.append(np.random.normal(0.0, pow(self.layerNodeCount[i], -0.5), (self.layerNodeCount[i+1], self.layerNodeCount[i])))
			print("W" + str(i),"weight matrix of size " + str(self.weights[len(self.weights) -1].shape))
		print("Output layer size: "+ str(self.layerNodeCount[i+1]))

	def accuracy(self, y):
		prob = np.copy(self.y_hat)
		prob[prob > 0.5] = 1
		prob[prob <= 0.5] = 0
		return (prob == y).all(axis=0).mean()

	def variance(self, y):
		return np.sum(np.square(y - self.y_hat))

	def forwardProp(self, X):
		self.layerOutput = [X]
		for i in range(0, len(self.layerNodeCount)-1):
			# Propagate through rest
			self.layerOutput.append(sigmoid(np.dot(self.layerOutput[i], self.weights[i].T)))
		self.y_hat = self.layerOutput[-1] # Set y_hat to the last (output) nodes


	def backProp(self, X, y):
		# Use Calulus chain rule to find derivative of the loss with respect to weights
		self.lastvariance = self.variance(y)
		# For each layer calculate the error
		for i in range(len(self.layerNodeCount)-2, -1, -1):
			if i == len(self.layerNodeCount)-2:
				errorsum = (2 * (y - self.y_hat) * derviativeSigmoid(self.y_hat)).T
			else:
				errorsum = (np.dot(self.weights[i+1].T, errorsum) * derviativeSigmoid(self.layerOutput[i+1]).T)

			self.delta = np.dot(errorsum, self.layerOutput[i])

			# Mutliplying by the learning rate
			self.delta = self.delta * self.lr

			# Nudging it
			self.weights[i] += self.delta

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
		lastOutput = sigmoid(np.dot(X, self.weights[0].T))
		# For each layer
		for i in range(1, len(self.layerNodeCount)-1):
				# Propagate through rest
				lastOutput = sigmoid(np.dot(lastOutput, self.weights[i].T))
		return lastOutput