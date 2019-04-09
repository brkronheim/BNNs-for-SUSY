import warnings
import network

import numpy as np

from BNN_functions import normalizeData
from layer import DenseLayer
from activationFunctions import Relu

warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Uncomment to use only CPUs
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Load data
trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)

#Normalize the training and output data and collect the values used to do so
normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 

#Network specs
outputDims=1
inputDims=19
hiddenDims=50
hiddenLayers=1
dtype=np.float32

#Setup network
neuralNet=network.network(dtype, inputDims, data[0], data[1], data[2], data[3])

#Add layers
seed=0
neuralNet.add(DenseLayer(inputDims,hiddenDims, seed=seed))
neuralNet.add(Relu())
seed+=1000
for n in range(hiddenLayers-1):
    neuralNet.add(DenseLayer(hiddenDims,hiddenDims, seed=seed))
    neuralNet.add(Relu())
    seed+=1000
neuralNet.add(DenseLayer(hiddenDims,outputDims, seed=seed))

neuralNet.setupMCMC(0.000010, 0.005, 20, 20) #Set up Markov Chains

#Train the network
savedResults=neuralNet.train(100,0,10, normInfo[0][0], normInfo[0][1],
                             folderName="1Deep50WideReluHMC",networksPerFile=10,
                             returnPredictions=False)
