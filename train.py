import ann
import sys
import numpy as np
import matplotlib.pyplot as plt


training_set="./mnist_train.csv" #short training set


"""Create instance of neural network """   
inputNodeCount = 784
hiddenNodesCount = [200]
outputNodeCount = 10
learningRate = 0.001
nn = ann.NeuralNetwork(inputNodeCount,hiddenNodesCount,outputNodeCount, learningRate)
nn.load('weights.npy')

"""Train the ANN"""
# load the mnist training data CSV file into a list
training_data_file = open(training_set, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 1000

print()
print("...TRAINING...")
for e in range(epochs):
	debug = 'Epochs: %i \tAccuracy: %f' % (e, (1-nn.lastvariance)*100) 
	print(debug, end="", flush=True)
	print('\r', end='', flush=True)
	
	for i in range(len(training_data_list)-1):

		record = training_data_list[i]

		try:    
			all_values = record.split(',') # split the record by the ',' commas
						
			inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #scale
			# create the target output values (all 0.01, except ans = 0.99)
			targets = np.zeros(outputNodeCount) + 0.01
			targets[int(all_values[0])] = 0.99 # all_values[0] answer for input
			X=np.array(inputs, ndmin=2) # create a 784 x 1 vector from input
			nn.train(X, targets)


		except KeyboardInterrupt:
			print('Interrupted')
			nn.save()
			exit(1)







nn.save()