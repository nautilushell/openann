import ann

import numpy as np

"""Create instance of neural network """   
learningRate = 0.001
nn = ann.NeuralNetwork("weights.npy", [784,200,10], "sigmoid", "quadratic")
#nn.setLearningRate(learningRate)
#nn.load('weights.npy')


"""Training set data and testing set data"""

test_set="./mnist_test_10.csv"       # short test set

"""Test ANN on new samples"""
test_data_file = open(test_set, 'r') # load the mnist test data CSV file into a list
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []
# go through all the records in the test data set
for record in test_data_list: 
	all_values = record.split(',') # split the record by the ',' commas
	correct_label = int(all_values[0])                        
	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs
	outputs = nn.guess(inputs) # ANN's guess as to the number
	label = np.argmax(outputs) # the index of the highest value corresponds to the answer
	if (label == correct_label):
		# network's answer matches correct answer, add 1 to scorecard
		print(correct_label, "correct")
		scorecard.append(1)
	else:

		# network's answer doesn't match correct answer, add 0 to scorecard
		print(correct_label, label, "wrong")
		scorecard.append(0)
		pass
	pass

"""Calculate the performance score, the fraction of correct answers"""
print (scorecard)
scorecard_array = np.asarray(scorecard)
print ("performance =", scorecard_array.sum() * 100./ scorecard_array.size, "%")