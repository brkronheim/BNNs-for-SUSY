import warnings
import os
import click

import numpy as np
import tensorflow as tf

from BNN_functions import normalizeData

from layer import DenseLayer
import network
from activationFunctions import Prelu, Relu

import time

start = time.time()

#This supresses many deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

#This tells tensorflow to only use CPUs. Assuming there are a lot of CPUs
#avaiable this may be faster than running on a GPU due to the parallel nature
#of this algorithm
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@click.command()
@click.option('--hidden', default=3, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--epochs', default=60, help='Number of epochs to train for')
@click.option('--burnin', default=0, help='Number of burnin epochs')
@click.option('--increment', default=50, help='Epochs between saving networks')
@click.option('--cores', default=4, help='Number of cores which can be used')
@click.option('--name', default="Default", help='Name of network')


def main(hidden, width, epochs, burnin, increment, cores, name):
    trainIn=np.load("trainInput.npy")
    trainOut=np.load("trainOutput.npy")
    trainOut=np.reshape(trainOut,(len(trainOut),1))

    valIn=np.load("validateInput.npy")
    valOut=np.load("validateOutput.npy")
    valOut=np.reshape(valOut,(len(valOut),1))

    testIn=np.load("testInput.npy")
    testOut=np.load("testOutput.npy")
    testOut=np.reshape(testOut,(len(testOut),1))
    
    #Normalize the training and output data and collect the values used to do so
    normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 
    
    #19 input values and 1 output
    outputDims=1
    inputDims=19
    
    dtype=np.float32
    
    #(datatype, input dimesions, training input, training output, validation input,
    #validation output, output mean, output sd)
    neuralNet=network.network(dtype, inputDims, data[0],data[1],data[2],data[3],normInfo[0][0], normInfo[0][1])
    
    #Add the layers
    seed=0
    neuralNet.add(DenseLayer(inputDims,width, seed=seed))
    #neuralNet.add(Prelu(width))
    neuralNet.add(Relu())
    seed+=1000
    for n in range(hidden-1):
        neuralNet.add(DenseLayer(width,width, seed=seed))
        #neuralNet.add(Prelu(width))
        neuralNet.add(Relu())
        seed+=1000
    neuralNet.add(DenseLayer(width,outputDims, seed=seed))
    
    #Setup the markov chain monte carlo
    neuralNet.setupMCMC(0.00001, 0.00001, 0.00003, 100, 20, 1000, 0.001, 20, burnin, cores)
    
    #Train the network
    neuralNet.train(epochs, burnin, increment, mean=normInfo[0][0], sd=normInfo[0][1],
                    scaleExp=True, folderName=name, networksPerFile=50, returnPredictions=False)
    
    end = time.time()
    print()
    print("elapsed time:", end - start)

if(__name__=="__main__"):
    main()
