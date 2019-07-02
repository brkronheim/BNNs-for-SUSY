import warnings
import os
import click

import numpy as np
import tensorflow as tf

from BNN_functions import normalizeData, trainBasic

from layer import DenseLayer
import network
from activationFunctions import Relu, Elu, Leaky_relu

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
@click.option('--nnCycles', default=2, help='Number of training cycles for normal neural net')
@click.option('--nnEpochs', default=30, help='Number of epochs per neural net cycle')
@click.option('--nnPatience', default=30, help='Early stopping patience for neural net')



def main(hidden, width, epochs, burnin, increment, cores, name, nnCycles, nnEpochs, patience):
    trainIn=np.load("trainInput.npy")
    trainOut=np.load("trainOutput.npy")
    trainOut=np.reshape(trainOut,(len(trainOut),1))

    valIn=np.load("validateInput.npy")
    valOut=np.load("validateOutput.npy")
    valOut=np.reshape(valOut,(len(valOut),1))

    #Normalize the training and output data and collect the values used to do so
    normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 
    
    #19 input values and 1 output
    outputDims=1
    inputDims=19
    
    dtype=np.float32
    weights=[None]*hidden
    biases=[None]*hidden
    if(cycles>0):
        weights, biases = trainBasic(hidden, inputDims, outputDims, width, nnCycles, nnEpochs, patience, data[0], data[1], data[2], data[3])
    
    
    
    #(datatype, input dimesions, training input, training output, validation input,
    #validation output, output mean, output sd)
    neuralNet=network.network(dtype, inputDims, data[0],data[1],data[2],data[3],normInfo[0][0], normInfo[0][1])
    
    #Add the layers
    seed=0
    neuralNet.add(DenseLayer(inputDims,width, weights=weights[0], biases=biases[0], seed=seed))
    #neuralNet.add(Prelu(width))
    neuralNet.add(Relu())
    seed+=1000
    for n in range(hidden-1):
        neuralNet.add(DenseLayer(width,width, weights=weights[n+1], biases=biases[n+1], seed=seed))
        #neuralNet.add(Prelu(width))
        neuralNet.add(Relu())
        seed+=1000
    neuralNet.add(DenseLayer(width,outputDims, weights=weights[-1], biases=biases[-1], seed=seed))
    
    #Setup the markov chain monte carlo
    neuralNet.setupMCMC(0.00001, 0.000005, 0.000015, 100, 20, 500, 0.00001, 20, burnin, cores)
    
    #Train the network
    neuralNet.train(epochs, burnin, increment, mean=normInfo[0][0], sd=normInfo[0][1],
                    scaleExp=True, folderName=name, networksPerFile=50, returnPredictions=False)
    
    end = time.time()
    print()
    print("elapsed time:", end - start)

if(__name__=="__main__"):
    main()
