from neuralnet import NeuralNet
import numpy as np
import utils
import math
import copy
import matplotlib.pyplot as plt


# net params    
NUMB_OF_NEURONS = 4
INPUT_SIZE = 8

# prepare some test data for code testing
m = 100
X_train = np.empty((INPUT_SIZE,m))
Y_train = np.empty((1,m))
X_train_1 = np.ones((INPUT_SIZE,int(m/2)))
Y_train_1 = np.ones((1,int(m/2)))
X_train_0 = np.random.rand(INPUT_SIZE,m-int(m/2))
Y_train_0 = np.zeros((1,m-int(m/2)))
X_train_conc = np.concatenate((X_train_0, X_train_1), axis=-1)   #axis=-1 corresponds to the last dimension
Y_train_conc = np.concatenate((Y_train_0, Y_train_1), axis=-1)   #axis=-1 corresponds to the last dimension
# shuffle data
subscripts = np.arange(m)
np.random.shuffle(subscripts)
for j in range(m):
    X_train[:,j] = X_train_conc[:,subscripts[j]]
    Y_train[:,j] = Y_train_conc[:,subscripts[j]]

    
    
# create and train neural network

nn = NeuralNet(input_size=INPUT_SIZE, neurons_num=NUMB_OF_NEURONS)
#nn.train(X_train, Y_train, epoch_num=100, batch_size=10, video_progress=False, cost_fun="cross_entropy", lambd=0.01)
nn.train(X_train, Y_train, epoch_num=100, batch_size=10, video_progress=False, cost_fun="cross_entropy", lambd=0.01)
          
