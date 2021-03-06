import numpy as np
import tensorflow as tf
import math

from scipy import stats

def normalizeData(trainIn, trainOut, valIn, valOut):
    """Normalizes the training and validation data to improve network training.
    
    The output data is normalized by taking its log and then scaling according
    to its normal distribution fit. The input data is normalized by scaling
    it down to [-1,1] using its min and max.
        
    Inputs:
        * trainIn: Numpy array containing the training input data
        * trainOut: Numpy array containing the training output data
        * valIn: Numpy array containing the validation input data
        * valOut: Numpy array containing the validation output data
        
    Returns:
        * data: List containing the normalized input data in the same order
        * normInfo: List containing the values required to un-normalize the data.
        *           Of the form: [(output_mean, output_sd), (input1_min, input1_max)
                                  (input2_min, input2_max), ...]
    """

    normInfo=[] #stores the data required to un-normalize the data
    
    #Take the log of the output distributions
    trainOutput=np.log(trainOut[:,1])
    valOutput=np.log(valOut[:,1])
    
    #Combine the output from the train and validation
    fullOutput=trainOutput.tolist()+valOutput.tolist()
    
    #Calculate the mean and standard deviation for the output
    mean, sd = stats.norm.fit(fullOutput)
    
    #Scale the output
    
    trainOutput-=mean
    trainOutput/=sd
    valOutput-=mean
    valOutput/=sd
    
    
    #Save the mean and standard deviation
    normInfo.append((mean,sd))

    #Scale all the input data from -1 to 1
    for x in range(len(trainIn[1,:])):
        minVal=min(np.amin(trainIn[:,x]),np.amin(valIn[:,x]))
        maxVal=max(np.amax(trainIn[:,x]),np.amax(valIn[:,x]))           
        trainIn[:,x]=(trainIn[:,x]-minVal)*2/(maxVal-minVal)-1
        valIn[:,x]=(valIn[:,x]-minVal)*2/(maxVal-minVal)-1
        
        #Save the min and max
        normInfo.append((minVal,maxVal))

    #Combine the data into a single list 
    data=[trainIn,trainOutput,valIn,valOutput]
    
    return(normInfo,data)

def multivariateLogProb(sigma, mu, x):
    """ Calculates the log probability of x given mu and sigma defining 
    a multivariate normal distribution. 
    
    Arguments:
        * sigma: an n-dimensional vector with the standard deviations of
        * the distribution
        * mu: an n-dimensional vector with the means of the distribution
        * x: m n-dimensional vectors to have their probabilities calculated
    Returns:
        * prob: an m-dimensional vector with the log-probabilities of x
    """
    
    sigma*=sigma
    sigma=tf.maximum(sigma, np.float32(10**(-16)))
    sigma=tf.minimum(sigma, np.float32(10**(16)))
    logDet=tf.reduce_sum(tf.log(sigma))
    k=tf.size(sigma, out_type=tf.float32)
    inv=tf.divide(tf.eye(k),sigma)
    dif=tf.subtract(x,mu)
    
    sigma=tf.linalg.diag(sigma)
    logLikelihood=-0.5*(logDet+tf.matmul(
            tf.matmul(tf.transpose(dif),inv),
            (dif))+k*tf.log(2*math.pi))   
    return(tf.linalg.diag_part(logLikelihood))

def loadNetworks(directoryPath):
    """Loads saved networks.
    
    Arguments:
        * directoryPath: the path to the directory where the networks are saved
    Returns:
        * numNetworks: Total number of networks 
        * numMatrices: Number of matrices in the network
        * matrices: A list containing all the extracted matrices
    
    """
    summary=[]
    with open(directoryPath+"summary.txt","r") as file:
        for line in iter(file):
            summary.append(line.split())
    
    numNetworks=int(summary[-1][0])
    numMatrices=int(summary[-1][2])
    numFiles=int(summary[-1][1])

    matrices=[]
    for n in range(numMatrices):
        weightsSplitDims=(numNetworks*numFiles,int(summary[n][0]),int(summary[n][1]))
        weights0=np.zeros(weightsSplitDims)
        for m in range(numFiles):
            weights=np.loadtxt(directoryPath+str(n)+"."+str(m)+".txt", dtype=np.float32,ndmin=2)
            print(weights.shape)
            print(weightsSplitDims)
            for k in range(numNetworks):
                weights0[m*numNetworks+k,:,:]=weights[weightsSplitDims[1]*k:weightsSplitDims[1]*(k+1),:weightsSplitDims[2]]
        matrices.append(weights0)
    numNetworks*=numFiles
    return(numNetworks, numMatrices, matrices)
    
def predict(inputMatrix, numNetworks, numMatrices, matrices):
    """Make predictions from an ensemble of neural networks. 
    
    Arguments:
        * inputMatrix: The input data
    Returns:
        * numNetworks: Number of networks used
        * numMatrices: Number of matrices in the network
        * matrices: List with all networks used
    """
    
    inputVal=np.transpose(inputMatrix)
    initialResults=[None]*(numNetworks)
    for m in range(numNetworks):
        current=inputVal
        for n in range(0,numMatrices,2):
            current=np.matmul(matrices[n][m,:,:],current)
            current+=matrices[n+1][m,:,:]
            if(n+2<numMatrices):
                current=np.maximum(current,0)
        if(m%100==0):
            print(m/numNetworks/numFiles)
        initialResults[m]=current
        
    return(initialResults)
