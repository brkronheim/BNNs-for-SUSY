import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from BNN_functions import multivariateLogProb

tfd = tfp.distributions

class DenseLayer(object):
    """Creates a 1 Dimensional Dense Bayesian Layer.
    
    Currently, the starting weight and bias mean values are 0.0 with a standard
    deviation of 1.0/sqrt(outputDims). The distribution that these values are 
    subject to have these values as their means, and a standard deviation of 
    2.0/sqrt(outputDims).
    """        
    def __init__(self, inputDims, outputDims, dtype=np.float32,seed=1):
        """
        Arguments:
            * inputDims: number of input dimensions
            * outputDims: number of output dimensions
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
        weightsMean=0.0
        self.weightsMeanHyper=tfd.MultivariateNormalDiag(loc=[weightsMean],
                    scale_diag=[.2])
        
        #weight SD value and SD distribution
        weightsSD=1/((self.outputDims)**(0.5))
        self.weightsSDHyper=tfd.MultivariateNormalDiag(loc=[weightsSD],
                    scale_diag=[1/((self.outputDims)**(0.5))])
        
        #bias mean value and mean distribution
        biasesMean=0.0
        self.biasesMeanHyper=tfd.MultivariateNormalDiag(loc=[biasesMean],
                    scale_diag=[.2])

        #bias SD value and SD distribution
        biasesSD=1/((self.outputDims)**(0.5))
        self.biasesSDHyper=tfd.MultivariateNormalDiag(loc=[biasesSD],
                    scale_diag=[2/((self.outputDims)**(0.5))])
        
        #Starting weight mean, weight SD, bias mean, and bias SD
        self.firstHypers=np.float32([weightsMean, weightsSD, biasesMean, biasesSD])

            
        #Placeholders for the weights, biases, and their means and SDs for use in HMC
        self.weights_chain_start = tf.placeholder(dtype, shape=(self.outputDims, self.inputDims))
        self.bias_chain_start = tf.placeholder(dtype, shape=(self.outputDims, 1))
        
        self.chains=[self.weights_chain_start, self.bias_chain_start]
        
        self.weight_mean_chain_start = tf.placeholder(dtype, shape=())
        self.weight_SD_chain_start = tf.placeholder(dtype, shape=())
        
        self.bias_mean_chain_start = tf.placeholder(dtype, shape=())
        self.bias_SD_chain_start = tf.placeholder(dtype, shape=())
        
        self.hyper_chains=[self.weight_mean_chain_start, self.weight_SD_chain_start,
                      self.bias_mean_chain_start, self.bias_SD_chain_start]
    
        
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
        weightsMean=self.hyper_chains[0]*tf.ones(shape=(self.outputDims,1))
        weightsSD=self.hyper_chains[1]*tf.ones(shape=(self.outputDims))
        biasesMean=self.hyper_chains[2]*tf.ones(shape=(self.outputDims,1))
        biasesSD=self.hyper_chains[3]*tf.ones(shape=(self.outputDims))
        prob=0

        #Calculate the probability of the paramaters given the current hypers
        val=multivariateLogProb(weightsSD,weightsMean,weights)/(self.inputDims*self.outputDims)
        prob+=tf.reduce_sum(val)
        val=multivariateLogProb(biasesSD,biasesMean,biases)/(self.outputDims)
        prob+=tf.reduce_sum(val)
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
        weightsMean=hypers[0]
        weightsSD=hypers[1]
        biasesMean=hypers[2]
        biasesSD=hypers[3]
        weights=self.weights_chain_start
        biases=self.bias_chain_start
        prob=0

        #Calculate probability of new hypers
        val=self.weightsMeanHyper.log_prob([[weightsMean]])
        prob+=tf.reduce_sum(val)
        val=self.weightsSDHyper.log_prob([[weightsSD]])
        prob+=tf.reduce_sum(val)

        val=self.biasesMeanHyper.log_prob([[biasesMean]])
        prob+=tf.reduce_sum(val)
        val=self.biasesSDHyper.log_prob([[biasesSD]])
        prob+=tf.reduce_sum(val)

        #Create tensors for prob calculation
        weightsMean*=tf.ones(shape=(self.outputDims,1))
        weightsSD*=tf.ones(shape=(self.outputDims))
        biasesMean*=tf.ones(shape=(self.outputDims,1))
        biasesSD*=tf.ones(shape=(self.outputDims))

        #Calculate probability of weights and biases given new hypers
        val=multivariateLogProb(weightsSD,weightsMean,weights)/(self.inputDims*self.outputDims)
        prob+=tf.reduce_sum(val)
        val=multivariateLogProb(biasesSD,biasesMean,biases)/(self.outputDims)
        prob+=tf.reduce_sum(val)

        return(prob)
        
    def sample(self):
        """Creates randomized weight and bias tensors based off 
        of their distributions
        
        Returns:
            * tempWeights: randomized weight tensor
            * tempBiases: randomized bias tensor
        """
        
        tempWeights = tf.random_normal((self.outputDims, self.inputDims),
                                       mean=self.firstHypers[0],
                                       stddev=self.firstHypers[1])
        tempBiases = tf.random_normal((self.outputDims, 1),
                                       mean=self.firstHypers[2],
                                       stddev=self.firstHypers[3])
        
        return(tempWeights, tempBiases)

    def expand(self, current):
        """Expands tensors to that they are of rank 2
        
        Arguments:
            * current: tensor to expand
        Returns:
            * expanded: expanded tensor
        
        """
        currentShape=tf.pad(
                tf.shape(current),
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
