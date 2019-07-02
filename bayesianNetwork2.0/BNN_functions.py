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
    
    print(trainOut.shape)
    
    #Take the log of the output distributions
    trainOutput=np.log(trainOut[:,0])
    valOutput=np.log(valOut[:,0])
    
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
    logDet=tf.reduce_sum(input_tensor=tf.math.log(sigma))
    k=tf.size(input=sigma, out_type=tf.float32)
    inv=tf.divide(tf.eye(k),sigma)
    dif=tf.subtract(x,mu)
    
    sigma=tf.linalg.diag(sigma)
    logLikelihood=-0.5*(logDet+tf.matmul(
            tf.matmul(tf.transpose(a=dif),inv),
            (dif))+k*tf.math.log(2*math.pi))   
    return(tf.linalg.diag_part(logLikelihood))

@tf.function
def cauchyLogProb(gamma, x0, x):
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
    a=tf.math.log(1+((x-x0)/gamma)**2)
    b1=math.pi*gamma
    b2=tf.cast(b1,tf.float32)
    b=tf.math.log(b2)
    c=tf.ones_like(x)
    d=-tf.math.scalar_mul(b,c)
    #b=-tf.math.scalar_mul(tf.math.log(np.float32(math.pi*gamma)),tf.ones_like(x))
    prob=a+d
    return(prob)


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

def trainBasic(hidden, inputDims, outputDims, width, cycles, epochs, patience, trainIn, trainOut, valIn, valOut):
    """Trains a basic neural network and returns its weights. Uses the Nadam optimizer and a learning rate of 
    0.01 which decays by a factor of 10 each cycle.
    
    Arguments:
        * hidden: number of hidden layers
        * inputDims: input dimension
        * outputDims: output dimension
        * width: width of hidden layers
        * cycles: number of training cycles with decaying learning rates
        * epochs: number of epochs per cycle
        * patience: early stopping patience
        * trainIn: training input data
        * trainOut: training output data
        * valIn: validation input data
        * valOut: validation output data
    Returns:
        * weights: list containing all weight matrices
        * biases: list containing all bias vectors
    """
    tf.random.set_seed(1000)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(width, kernel_regularizer=tf.keras.regularizers.l1_l2(0.0,0.0), kernel_initializer='glorot_uniform', input_shape=(inputDims, ), activation='relu'))
    for n in range(hidden-1):
        model.add(tf.keras.layers.Dense(width, kernel_regularizer=tf.keras.regularizers.l1_l2(0.0,0.0), kernel_initializer='glorot_uniform', activation='relu'))
    model.add(tf.keras.layers.Dense(outputDims, kernel_regularizer=tf.keras.regularizers.l1_l2(0.0,0.0),  kernel_initializer='glorot_uniform', activation='linear'))
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    for x in range(cycles):

        model.compile(optimizer=tf.keras.optimizers.Nadam(0.001*(10**(-x))),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])
        model.summary()
        model.fit(trainIn, trainOut, validation_data=(valIn, valOut), epochs=epochs, batch_size=32, callbacks=[callback])
    
    
    
    weights=[]
    biases=[]
    for layer in model.layers:
        weightBias=layer.get_weights()
        if(len(weightBias)==2):
            weights.append(weightBias[0].T)
            bias=weightBias[1]
            bias=np.reshape(bias, (len(bias),1))
            biases.append(bias)
    return(weights, biases)
    
