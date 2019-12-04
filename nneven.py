#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Jim
"""

import numpy as np
import random

def sigmoid(x):
        return 1/(1+np.exp(-x))
    
def derviativeSigmoid(x):
    return x * (1-x)

"""ANN class"""
class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.lr = learningRate
        
        # weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node j to node i in the next layer
        # wih = 200 x 784 matrix for weights for layer between input -> hidden
        # who = 10 x 200 matrix for weights for layer between hidden -> output
        # random weights in normal distribution around 0.0 with std deviation of  1/sqr(number of nuerons layer)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        pass
     
    def forwardProp(self, X):
        self.layer1 = sigmoid(np.dot(X, self.wih.T))  # save hidden layer activation calculation
        self.y_hat = sigmoid(np.dot(self.layer1, self.who.T)) # save ANN's hypothesis
        pass
        
    def backProp(self, X, y):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        tempCalc = 2*(y - self.y_hat) * derviativeSigmoid(self.y_hat)
        d_who = self.lr * np.dot(self.layer1.T, tempCalc)
        d_wih = self.lr * np.dot(X.T, (np.dot(tempCalc, self.who) * derviativeSigmoid(self.layer1)))
       
        self.wih += d_wih.T
        self.who += d_who.T
                
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


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)
     

"""Create instance of neural network"""  
bitlength=16
print("This ANN determines whether counting number is even or odd from 0 -",2**bitlength-1)
inputNodeCount = bitlength
hiddenNodeCount = 40
outputNodeCount = 2
learningRate = 0.1
nn = neuralNetwork(inputNodeCount,hiddenNodeCount,outputNodeCount, learningRate)
trainingcount=100

"""Create a list of random number from 0 - 2^bitlength - 1"""
num=[]
for i in range(1,trainingcount):
    numx=random.randint(0,2**bitlength-1)
    num.extend([numx])
 
""" Train network"""
epochs = 5
for e in range(epochs): 
    for i in range(1,trainingcount):
        inputs = np.array(bin_array(num[i-1],bitlength)) 
        targets = np.zeros(outputNodeCount)
        targets[num[i-1]%2] = 1
        X=np.array(inputs, ndmin=2) # create input array for ANN
        nn.train(X, targets)
print("ANN finished training")

"""Test network"""
print("Testing ANN with new random numbers")
scorecard=[]
for i in range(1,25): 
    num=random.randint(0,2**bitlength-1)
    bin_num = bin_array(num,bitlength)                       
    inputs = np.array(bin_num) 
    outputs = nn.guess(inputs)  # see what the network says is the correct answer
    indextoans = np.argmax(outputs) # the index of the highest value corresponds to the answer
#    print(num,outputs, indextoans, inputs)
    if (indextoans == 1 and num%2 == 1):
        # network's answer matches correct answer, add 1 to scorecard
        print(num, "odd")
        scorecard.append(1)
    elif indextoans == 0 and num%2==0:
        # network's answer doesn't match correct answer, add 0 to scorecard
        print(num, "even")
        scorecard.append(1)
    else:
        print(num, "incorrect")
        scorecard.append(0)
        pass
    pass
  
"""Calculate the performance score, the fraction of correct answers"""
print ("correct answers:",scorecard)
scorecard_array = np.asarray(scorecard)
print ("performance =", scorecard_array.sum() * 100./ scorecard_array.size, "%")