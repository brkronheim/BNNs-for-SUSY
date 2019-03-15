import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class DenseLayer(object):
    """Creates a 1 Dimensional Dense Bayesian Layer."""        
    def __init__(self, inputDims, outputDims, dtype=np.float32):
        """
        Arguments:
            * inputDims: number of input dimensions
            * outputDims: number of output dimensions
            * dtype: data type of input and output values
        """
        
        self.inputDims=inputDims
        self.outputDims=outputDims
        self.dtype=dtype
        
        self.allWeights=[None]*outputDims #List to store all weight distributions
        
        #Weight mean value and mean distribution
        self.weightsMean=tf.Variable(0, dtype=self.dtype)
        self.weightsMeanHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        self.weightsMeanHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        
        #weight SD value and SD distribution
        self.weightsSD=tf.Variable(1, dtype=self.dtype)
        self.weightsSDHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsSD],
                    scale_diag=[1.0])
        self.weightsSDHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])

        self.allBiases=[None]*outputDims #List to store all bias distributions
        
        #bias mean value and mean distribution
        self.biasesMean=tf.Variable(0, dtype=self.dtype)
        self.biasesMeanHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        self.biasesMeanHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])

        #bias SD value and SD distribution
        self.biasesSD=tf.Variable(1, dtype=self.dtype)
        self.biasesSDHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsSD],
                    scale_diag=[1.0])
        self.biasesSDHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        
        #Starting weight mean, weight SD, bias mean, and bias SD
        self.firstHypers=np.float32([0.0, 1.0, 0.0, 1.0])

        #Create the weight and bias distributions for the perceptrons in a layer
        for n in range(self.outputDims):
            self.allWeights[n], self.allBiases[n] = self.createPerecptron(
                    self.weightsMean, self.weightsSD, self.biasesMean, 
                    self.biasesSD, self.inputDims, self.dtype)
            
        #Placeholders for the weights, biases, and their means and SDs for use in HMC
        self.weights_chain_start = tf.placeholder(dtype, shape=(self.outputDims, self.inputDims))
        self.bias_chain_start = tf.placeholder(dtype, shape=(self.outputDims, 1))
        
        self.weight_mean_chain_start = tf.placeholder(dtype, shape=())
        self.weight_SD_chain_start = tf.placeholder(dtype, shape=())
        
        self.bias_mean_chain_start = tf.placeholder(dtype, shape=())
        self.bias_SD_chain_start = tf.placeholder(dtype, shape=())
        
    
    def createPerceptron(self):
        """Creates the weight and biases distributions for a single perceptron"""
        weights = tfd.MultivariateNormalDiag(
                loc=np.ones(self.inputDims, dtype=self.dtype)*self.weightsMean,
                scale_diag=np.ones(self.inputDims, dtype=self.dtype)*self.weightsSD)

        bias = tfd.MultivariateNormalDiag(
                loc=[self.biasesMean],
                scale_diag=[self.biasesSD])
        
        return(weights, bias)
        
        
    def update(self, weightsMean, weightsSD, biasesMean, biasesSD):
        """Updates the weight and bias distributions for the layer when their
        means and SDs change.
        """
        self.weightsMean=tf.Variable(weightsMean, dtype=self.dtype)
        self.weightsSD=tf.Variable(weightsSD, dtype=self.dtype)
        self.biasesMean=tf.Variable(biasesMean, dtype=self.dtype)
        self.biasesSD=tf.Variable(biasesSD, dtype=self.dtype)
        for n in range(self.outputDims):
            self.allWeights[n], self.allBiases[n] = self.createPerecptron(
                    self.weightsMean, self.weightsSD, self.biasesMean, 
                    self.biasesSD, self.inputDims, self.dtype)  
