import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from BNN_functions import multivariateLogProb, cauchyLogProb

tfd = tfp.distributions

class DenseLayer(object):
    """Creates a 1 Dimensional Dense Bayesian Layer.
    
    Currently, the starting weight and bias mean values are 0.0 with a standard
    deviation of 1.0/sqrt(outputDims). The distribution that these values are 
    subject to have these values as their means, and a standard deviation of 
    2.0/sqrt(outputDims).
    """        
    def __init__(self, inputDims, outputDims, weights=None, biases=None, dtype=np.float32,seed=1):
        """
        Arguments:
            * inputDims: number of input dimensions
            * outputDims: number of output dimensions
            * weights: list of starting weight matrices
            * biases: list of starting bias vectors
            * dtype: data type of input and output values
            * seed: seed used for random numbers
        """
        self.numTensors=2 #Number of tensors used for predictions
        self.numHyperTensors=4 #Number of tensor for hyper paramaters
        self.inputDims=inputDims
        self.outputDims=outputDims
        self.dtype=dtype
        self.seed=seed
            
        #Weight mean value and mean distribution
        weightsx0=0.0
        self.weightsx0Hyper=tfd.MultivariateNormalDiag(loc=[weightsx0],
                    scale_diag=[.2])
        
        #weight SD value and SD distribution
        weightsGamma=0.5
        self.weightsGammaHyper=tfd.MultivariateNormalDiag(loc=[weightsGamma],
                    scale_diag=[0.5])
        
        #bias mean value and mean distribution
        biasesx0=0.0
        self.biasesx0Hyper=tfd.MultivariateNormalDiag(loc=[biasesx0],
                    scale_diag=[.2])

        #bias SD value and SD distribution
        biasesGamma=0.5
        self.biasesGammaHyper=tfd.MultivariateNormalDiag(loc=[biasesGamma],
                    scale_diag=[0.5])
        
        #Starting weight mean, weight SD, bias mean, and bias SD
        self.hypers=np.float32([weightsx0, weightsGamma, biasesx0, biasesGamma])

        #Starting weights and biases
        if(weights is None):
            self.parameters = self.sample()
        else:
            self.parameters=[weights, biases]
        
    def calculateProbs(self, weightBias):
        """Calculates the log probability of a set of weights and biases given
        their distributions in this layer.
        
        Arguments:
            * weightsBias: list with new possible weight and bias tensors 
            
        Returns:
            * prob: log prob of weights and biases given their distributions
        """
        #Create the tensors used to calculate probability
        weights=weightBias[0]
        biases=weightBias[1]
        weightsx0=self.hypers[0]#*tf.ones(shape=(self.outputDims,1))
        weightsGamma=self.hypers[1]#*tf.ones(shape=(self.outputDims))
        biasesx0=self.hypers[2]#*tf.ones(shape=(self.outputDims,1))
        biasesGamma=self.hypers[3]#*tf.ones(shape=(self.outputDims))
        prob=0

        #Calculate the probability of the paramaters given the current hypers
        val=cauchyLogProb(weightsGamma,weightsx0,weights)#/(self.inputDims*self.outputDims)
        prob+=tf.reduce_sum(input_tensor=val)
        val=cauchyLogProb(biasesGamma,biasesx0,biases)#/(self.outputDims)
        prob+=tf.reduce_sum(input_tensor=val)
        return(prob)
        
        
    def calculateHyperProbs(self, hypers, weightBias):
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
        weightsx0=hypers[0]
        weightsGamma=hypers[1]
        biasesx0=hypers[2]
        biasesGamma=hypers[3]
        weights=weightBias[0]
        biases=weightBias[1]
        prob=0

        #Calculate probability of new hypers
        val=self.weightsx0Hyper.log_prob([[weightsx0]])
        prob+=tf.reduce_sum(input_tensor=val)
        val=self.weightsGammaHyper.log_prob([[weightsGamma]])
        prob+=tf.reduce_sum(input_tensor=val)

        val=self.biasesx0Hyper.log_prob([[biasesx0]])
        prob+=tf.reduce_sum(input_tensor=val)
        val=self.biasesGammaHyper.log_prob([[biasesGamma]])
        prob+=tf.reduce_sum(input_tensor=val)

        #Create tensors for prob calculation
        #weightsx0*=tf.ones(shape=(self.outputDims,1))
        #weightsGamma*=tf.ones(shape=(self.outputDims))
        #biasesx0*=tf.ones(shape=(self.outputDims,1))
        #biasesGamma*=tf.ones(shape=(self.outputDims))

        #Calculate probability of weights and biases given new hypers
        val=cauchyLogProb(weightsGamma,weightsx0,weights)#/(self.inputDims*self.outputDims)
        prob+=tf.reduce_sum(input_tensor=val)
        val=cauchyLogProb(biasesGamma,biasesx0,biases)#/(self.outputDims)
        prob+=tf.reduce_sum(input_tensor=val)

        return(prob)
        
    def sample(self):
        """Creates randomized weight and bias tensors based off 
        of their distributions
        
        Returns:
            * tempWeights: randomized weight tensor
            * tempBiases: randomized bias tensor
        """
        
        tempWeights = tf.random.normal((self.outputDims, self.inputDims),
                                       mean=self.hypers[0],
                                       stddev=(2/self.outputDims)**(0.5),
                                      seed=self.seed)
        tempBiases = tf.random.normal((self.outputDims, 1),
                                       mean=self.hypers[2],
                                       stddev=(2/self.outputDims)**(0.5),
                                       seed=self.seed+1)
        
        return([tempWeights, tempBiases])

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
    
    def predict(self,inputTensor, weightBias):
        """Calculates the output of the layer based on the given input tensor
        and weight and bias values
        
        Arguments:
            * inputTensor: the input tensor the layer acts on
            * weightBias: a list with the current weight and bias tensors
        Returns:
            * result: the output of the layer
        
        """
        weightTensor=self.expand(weightBias[0])
        biasTensor=self.expand(weightBias[1])
        result=tf.add(tf.matmul(weightTensor, inputTensor), biasTensor)
        return(result)
    
    def updateParameters(self, weightBias):
        self.parameters = weightBias
        
    def updateHypers(self, hypers):
        self.hypers = hypers
