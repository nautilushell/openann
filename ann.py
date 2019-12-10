#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Julian Allchin
"""

import numpy as np

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def derviativeSigmoid(z):
     sig=sigmoid(z)
     return (sig * (1 - sig))


"""Artificial Neural Network Class"""
class NeuralNetwork:
    def __init__(self, NetSize, learningRate):
        
        self.layerNodeCount = NetSize
        self.numberOfLayers = len(NetSize)
        self.error = []
        self.lr = learningRate
        self.lastvariance = 100

        # Create the neural network biases and weights arrays
#        self.biases = [np.ones(i) for i in NetSize[1:]]
        self.biases = [np.zeros(i) for i in NetSize[1:]]
        self.weights=[np.random.randn(j, i)*0.1 for i, j in zip(NetSize[:-1], NetSize [1:])] 
        
        # Print out the sizes of the layers
        print("Input layer size: "+ str(NetSize[0]))
        for i in range(len(self.weights)):
            print("W" + str(i),"weight matrix of size " + str(self.weights[i].shape))
        print("Output layer size: "+ str(NetSize[-1]))

    def accuracy(self):
        return (1-self.lastvariance)

    def variance(self, y):
        return np.sum(np.square(y - self.y_hat))

    def forwardProp(self, X):
        self.layerOutput = [X]
        self.layerInput= [X]
        for i in range(self.numberOfLayers-1):
            self.layerInput.append(np.dot(self.layerOutput[i], self.weights[i].T)+ self.biases[i])
            self.layerOutput.append(sigmoid(self.layerInput[-1]))  
        self.y_hat = self.layerOutput[-1] # Set y_hat to the last (output) nodes


    def backProp(self, X, y):
        # Use Calulus chain rule to find derivative of the loss with respect to weights
        self.lastvariance = self.variance(y)
        # For each layer calculate the error
        for i in range(self.numberOfLayers-2, -1, -1):
            if i == self.numberOfLayers-2:
                errorsum = (2 * (y - self.y_hat) * derviativeSigmoid(self.layerInput[-1])).T
            else:
                errorsum = (np.dot(self.weights[i+1].T, errorsum) * derviativeSigmoid(self.layerInput[i+1]).T)

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
    def save(self, fname):
        with open(fname,'wb') as f:
            np.save(f, self.weights)
        f.close()
        print("Saved weights")

    # after training save weights
    def load(self, file):
        with open(file, 'rb') as f:
            self.weights = np.load(f)
        f.close()

    def guess(self, X):
        a = X
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) +b)
#        print (a)
        return a