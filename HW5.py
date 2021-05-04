from NeuralNetworks.neuralNetwork import *
from NeuralNetworks.torchnn import *
import numpy as np

training = "Perceptron/bank-note/train.csv"
testing = "Perceptron/bank-note/test.csv"


print("RELU")
network = torchnn()
network.train(training,2)
network.test(testing)


print("TANH")
network = torchnn(tanh=True)
network.train(training,2)
network.test(testing)