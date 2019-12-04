#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Jim Allchin
"""

import numpy as np
import matplotlib.pyplot as plt

# files for training & testing: mnist_train_100 is 100 samples; mnist_train is all
training_set="./mnist_train.csv" 
test_set="./mnist_test_10.csv"

# number of images to show during training
dispimages=100
 
trained=0                   # set to 1 to use previously saved weights
savetrainingweights=0       # set to 1 to save weights after training
  

"""activation function and its derviative"""
def sigmoid(x):
		return 1/(1+np.exp(-x))
	
def derviativeSigmoid(x):
	return x * (1-x)

"""Artificial Neural Network Class"""
class neuralNetwork:
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		self.inodes = inputNodes
		self.hnodes = hiddenNodes
		self.onodes = outputNodes
		self.lr = learningRate
		self.error = []
		
		# weight matrices: wih and who
		# weight arrays are w_i_j: link goes from node j to node i in next layer
		# wih = 200 x 784 matrix for weights for layer between input -> hidden
		# who = 10 x 200 matrix for weights for layer between hidden -> output
		# random weights in normal distribution around 0.0 with std deviation
		#   of 1/sqr(number of neurons layer)
		self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), 
									(self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), 
									(self.onodes, self.hnodes))
		pass
	 
	def forwardProp(self, X):
		self.layer1 = sigmoid(np.dot(X, self.wih.T))  # save hidden layer weight
		self.y_hat = sigmoid(np.dot(self.layer1, self.who.T)) # save ANN's ans
		pass
		
	def backProp(self, X, y):
		# application of the chain rule to find derivative of the loss 
		#  function with respect to weights2 and weights1
		tempCalc = 2*(y - self.y_hat) * derviativeSigmoid(self.y_hat)
		d_who = self.lr * np.dot(self.layer1.T, tempCalc)
		d_wih = self.lr * np.dot(X.T, (np.dot(tempCalc, self.who) * 
									   derviativeSigmoid(self.layer1)))
	   
		self.wih += d_wih.T
		self.who += d_who.T
				
	# train the neural network
	def train(self, X, y):
		self.forwardProp(X)
		self.error.append(np.sum(np.square(y - self.y_hat)))
		self.backProp(X, y)
		
	# after training save weights
	def saveANN(self):
		with open('wih.txt','wb') as f:        #save wih
				np.savetxt(f, self.wih, delimiter=',')
		f.close()
		with open('who.txt','wb') as f:         #save who
				np.savetxt(f, self.who ,delimiter=',')
		f.close()
		
	# restore training saved weights
	def returnANN(self):
		with open('wih.txt','r') as f:
				self.wih=np.loadtxt(f, delimiter=',')
		f.close()
		with open('who.txt','r') as f:
				self.who=np.loadtxt(f, delimiter=',')
		f.close()
		
	# ANN's guess  
	def guess(self, X):
		 # calculate signals to hidden layer: (200 x 784)*(784 x 1) => (200 x 1)
		hiddenOut = sigmoid(np.dot(self.wih, X))
		# calculate signals to output layer (10 x 200)*(200 x 1) => (10 x 1)
		finalOut = sigmoid(np.dot(self.who, hiddenOut))
		return finalOut


 
"""Create ANN"""
inputNodeCount = 784
hiddenNodeCount = 200
outputNodeCount = 10
learningRate = 0.005
trainstopError = 0.5
nn = neuralNetwork(inputNodeCount,hiddenNodeCount,outputNodeCount, learningRate)


"""Train ANN"""
if (trained !=1):
	print("Training ANN")
	training_data_file = open(training_set, 'r') # load mnist training CSV file
	training_data_list = training_data_file.readlines()
	training_data_file.close()
	epochs = 5
	displaycount=0
	accuracy = [0]
	numcorrect = 0
	complete = 0
	# for e in range(epochs):
	try:
		for record in training_data_list:  
			complete = complete + 1    
			all_values = record.split(',') # split the record by the ',' commas
			
			inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #scale
			# create the target output values (all 0.01, except ans = 0.99)
			targets = np.zeros(outputNodeCount) + 0.01
			targets[int(all_values[0])] = 0.99 # all_values[0] answer for input
			X=np.array(inputs, ndmin=2) # create a 784 x 1 vector from input
			nn.train(X, targets)

			correct = nn.guess(X.T)
			label = np.argmax(correct) # index to highest value is answer

			if (label == int(all_values[0])):
				numcorrect = numcorrect + 1
				accuracy.append(numcorrect/complete)

			
			plt.clf()

			# axes = plt.gca()
			# axes.set_ylim([0,1])
			plt.plot(nn.error)
			plt.draw()
			plt.pause(.001)
			# print ("accuracy: %s "  % (accuracy[len(accuracy) - 1]))
			# 
			
	
	except KeyboardInterrupt:
		print('Interrupted')
		nn.saveANN()

	if (savetrainingweights == 1):
		nn.saveANN()
else:
	nn.returnANN()

"""Test ANN with new data set"""
print("ANN testing new handwritten dataset")
test_data_file = open(test_set, 'r') # load the mnist test data CSV file
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []

for record in test_data_list:   # go through all the records in the test set
	all_values = record.split(',') # split the record by the ',' commas
	correct_label = int(all_values[0])                        
	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # scale
	outputs = nn.guess(inputs) # see what ANN says the digit is
	label = np.argmax(outputs) # index to highest value is answer
	if (label == correct_label):
		print(correct_label, "correct")
		scorecard.append(1)
	else:
		print(correct_label, "wrong")
		scorecard.append(0)
		pass
	pass
  
"""Calculate the performance score, the fraction of correct answers"""
print ("correct answers: ",scorecard)
scorecard_array = np.asarray(scorecard)
print ("performance =", scorecard_array.sum() * 100./ scorecard_array.size, "%")


