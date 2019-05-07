import warnings
import os
import click

import numpy as np

from BNN_functions import normalizeData

from layer import DenseLayer
import network
from activationFunctions import Relu

#This supresses many deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
    trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
    trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
    valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
    valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)
    
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
    neuralNet.add(Relu())
    seed+=1000
    for n in range(hidden-1):
        neuralNet.add(DenseLayer(width,width, seed=seed))
        neuralNet.add(Relu())
        seed+=1000
    neuralNet.add(DenseLayer(width,outputDims, seed=seed))
    
    #Setup the markov chain monte carlo
    neuralNet.setupMCMC(0.00005, 0.000001, 0.00005, 100, 20, 1000, 0.001, 20, burnin, cores)
    
    #Train the network
    neuralNet.train(epochs, burnin, increment, normInfo[0][0], normInfo[0][1],
                    folderName=name, networksPerFile=50, returnPredictions=False)

if(__name__=="__main__"):
    main()
