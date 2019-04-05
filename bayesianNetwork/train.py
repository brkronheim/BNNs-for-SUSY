import warnings
import os

import numpy as np

from scipy import stats
from BNN_functions import normalizeData

from layer import DenseLayer
import network
from activationFunctions import Relu

warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)



#Normalize the training and output data and collect the values used to do so
normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 
outputDims=1
inputDims=19
hiddenDims=50
hiddenLayers=3
dtype=np.float32

weights0=np.loadtxt("A40weights0.txt",dtype=dtype)
weights1=np.loadtxt("A40weights1.txt",dtype=dtype)
weights2=np.loadtxt("A40weights2.txt",dtype=dtype)
weights3=np.loadtxt("A40weights3.txt",dtype=dtype)
biases0=np.loadtxt("A40biases0.txt",dtype=dtype)
biases1=np.loadtxt("A40biases1.txt",dtype=dtype)
biases2=np.loadtxt("A40biases2.txt",dtype=dtype)
biases3=np.loadtxt("A40biases3.txt",dtype=dtype)



neuralNet=network.network(np.float32, data[0],data[1],data[2],data[3])

neuralNet.add(DenseLayer(inputDims,hiddenDims, seed=0), weights=np.transpose(weights0), 
              biases=np.reshape(biases0,(hiddenDims,1)))
neuralNet.add(Relu())
neuralNet.add(DenseLayer(hiddenDims,hiddenDims, seed=1000), weights=np.transpose(weights1), 
              biases=np.reshape(biases1,(hiddenDims,1)))
neuralNet.add(Relu())
neuralNet.add(DenseLayer(hiddenDims,hiddenDims, seed=2000), weights=np.transpose(weights2), 
              biases=np.reshape(biases2,(hiddenDims,1)))
neuralNet.add(Relu())
neuralNet.add(DenseLayer(hiddenDims,outputDims, seed=3000), weights=np.transpose(np.reshape(weights3,(50,1))), 
              biases=np.reshape(biases3,(1,1)))


neuralNet.setupMCMC(0.00001, 0.0006, 20, 20)
savedResults=neuralNet.train(5000,0,1,10,0,5, normInfo[0][0], normInfo[0][1],folderName="3Deep50WideReluHMC")
