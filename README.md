# openann

An Artificial Neural Network written in python only using the linear algebra tool [NumPy](https://numpy.org/). It was created as an experiment of a simple, diverse ANN which was user friendly and concise.

### Usage

```python
import ann

# Initialization
nodes = [781, 50, 30,..., 10] # input, hidden1, hidden2..., output
lr = 0.001 # Learning rate
nn = ann.NeuralNetwork(nodes, learningRate) # Network with nodes and LR
nn.load('weights.txt') # Loads the previous weights

# Train
for i in training_data.range():
	nn.train(i.inputs, i.correctOutput) # Train
	print(nn.accuracy)

nn.save('weights.txt')

# Guess
inputs = test_file.readlines() # Simplified
outputs = nn.guess(inputs)

```

