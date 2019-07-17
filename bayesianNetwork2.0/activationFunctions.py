from tensorflow.python.ops import gen_nn_ops
import tensorflow.nn as nn
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd=tfp.distributions

class Relu(object):
    """Relu activation function"""        
    def __init__(self):
        self.numTensors=0
        self.numHyperTensors=0

    def predict(self,inputTensor,_):
        result=gen_nn_ops.relu(inputTensor)
        return(result)
    
class Sigmoid(object):
    """Sigmoid activation function"""        
    def __init__(self):
        self.numTensors=0
        self.numHyperTensors=0

    def predict(self,inputTensor,_):
        result=tf.math.sigmoid(inputTensor)
        return(result)
    
class Tanh(object):
    """Tanh activation function"""        
    def __init__(self):
        self.numTensors=0
        self.numHyperTensors=0

    def predict(self,inputTensor,_):
        result=tf.math.tanh(inputTensor)
        return(result)
        
class Elu(object):
    """Elu activation function"""        
    def __init__(self):
        self.numTensors=0
        self.numHyperTensors=0

    def predict(self,inputTensor,_):
        result=gen_nn_ops.elu(inputTensor)
        return(result)
        

class Softmax(object):
    """Softmax activation function"""        
    def __init__(self):
        self.numTensors=0
        self.numHyperTensors=0

    def predict(self,inputTensor, _):
        result=gen_nn_ops.softmax(inputTensor)
        return(result)
    
class Leaky_relu(object):
    """Leaky relu activation function"""
    def __init__(self, alpha=0.3):
        self.numTensors=0
        self.numHyperTensors=0
        self.alpha=alpha
    def predict(self, inputTensor, _):
        result=nn.leaky_relu(inputTensor, self.alpha)
        return(result)
    
class Prelu(object):
    """Prelu activation function"""
     
    def __init__(self, inputDims, dtype=np.float32, alpha=0.2, activation=None, seed=1):
        """
        Arguments:
            * inputDims: number of input dimensions
            * dtype: data type of input and output values
            * alpha: Single custom starting slope value
            * activation: optional custom values for starting slope values
            * seed: seed used for random numbers
        """
        self.numTensors=1 #Number of tensors used for predictions
        self.numHyperTensors=1 #Number of tensor for hyper paramaters
        self.inputDims=inputDims
        self.dtype=dtype
        self.seed=seed
            
        #Starting rate value and hyperRate
        rate=0.3
        self.hyperRate=0.3
        
        #Starting weight mean, weight SD, bias mean, and bias SD
        
        self.hypers=np.float32([rate])

        #Starting weights and biases
        if(activation is None):
            self.parameters = [alpha*tf.ones(shape=(inputDims))]    
        else:
            self.parameters = [activation]
        #print(self.parameters)
        
    @tf.function
    def exponentialLogProb(self, rate, x):
        """Calcualtes the log probability of an exponential distribution.
        
        Arguments:
            * rate: rate parameter for the distribution
            * x: input value
        Returns:
            * logProb: log probability of x
        """
        
        rate=tf.absolute(rate)
        logProb=-rate*x+tf.math.log(rate)        
        
        return(logProb)
        
    @tf.function
    def calculateProbs(self, slopes):
        """Calculates the log probability of the slopes given
        their distributions in this layer.
        
        Arguments:
            * weightsBias: list with new possible weight and bias tensors 
            
        Returns:
            * prob: log prob of weights and biases given their distributions
        """
        
        val=self.exponentialLogProb(self.hypers[0],slopes)
        prob=tf.reduce_sum(input_tensor=val)
        
        return(prob)
        
    @tf.function
    def calculateHyperProbs(self, hypers, slopes):
        """Calculates the log probability of a set of weights and biases given
        new distribtuions as well as the probability of the new distribution
        means and SDs given their distribtuions.
        
        Arguments:
            * hypers: a list containg 4 new possible hyper parameters
            * weightBias: a list with the current weight and bias matrices
            
        Returns:
            * prob: log probability of weights and biases given the new hyper parameters 
            and the probability of the new hyper parameters given their priors
        """
        
        rate=tf.maximum(hypers[0],0.01)
        slopes=tf.maximum(slopes[0],0)
        prob=0

        #Calculate probability of new hypers
        val=self.exponentialLogProb(self.hyperRate, hypers[0])
        prob+=tf.reduce_sum(input_tensor=val)

        #Calculate probability of weights and biases given new hypers
        val=self.exponentialLogProb(hypers[0],slopes)
        prob+=tf.reduce_sum(input_tensor=val)

        return(prob)
        
    @tf.function
    def expand(self, current):
        """Expands tensors to that they are of rank 2
        
        Arguments:
            * current: tensor to expand
        Returns:
            * expanded: expanded tensor
        
        """
        currentShape=tf.pad(
                tensor=tf.shape(input=current),
                paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
                constant_values=1)
        expanded=tf.reshape(current, currentShape)
        return(expanded)
    
    @tf.function
    def predict(self,inputTensor, slopes):
        """Calculates the output of the layer based on the given input tensor
        and weight and bias values
        
        Arguments:
            * inputTensor: the input tensor the layer acts on
            * weightBias: a list with the current weight and bias tensors
        Returns:
            * result: the output of the layer
        
        """
        slopes=slopes[0]
        
        slopes=tf.reshape(slopes, (len(slopes),1))
        activated=tf.multiply(slopes, inputTensor)
        result=tf.where(tf.math.less(inputTensor,0), activated, inputTensor)
        return(self.expand(result))
    
    def updateParameters(self, slopes):
        self.parameters = [tf.maximum(0,slopes[0])]
        
    def updateHypers(self, hypers):
        self.hypers = [tf.maximum(0.01,hypers[0])]
