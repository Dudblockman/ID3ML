from NeuralNetworks.neuralNetwork import *
from NeuralNetworks.torchnn import *
import numpy as np

training = "Perceptron/bank-note/train.csv"
testing = "Perceptron/bank-note/test.csv"

widths = [5,10,25,50,100]
depths = [3,5,9]

reluResults = np.zeros((len(widths), len(depths)))
tanhResults = np.zeros((len(widths), len(depths)))
for i in range(len(widths)):
    for j in range(len(depths)):
        network = torchnn(width=widths[i], depth=depths[j])
        network.train(training,2)
        tanhResults[i][j] = network.test(testing)

        network = torchnn(width=widths[i], depth=depths[j],tanh=True)
        network.train(training,2)
        tanhResults[i][j] = network.test(testing)
        
print("RELU")
print(reluResults)
print("TANH")
print(tanhResults)