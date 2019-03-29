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
        self.numTensors=2
        self.numHyperTensors=4
        self.inputDims=inputDims
        self.outputDims=outputDims
        self.dtype=dtype
        
        self.allWeights=[None]*outputDims #List to store all weight distributions
        
            
        #Weight mean value and mean distribution
        self.weightsMean=tf.Variable(0, dtype=self.dtype)
        self.weightsMeanHyper=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[2.0])
        #weight SD value and SD distribution
        self.weightsSD=tf.Variable(1, dtype=self.dtype)
        self.weightsSDHyper=tfd.MultivariateNormalDiag(loc=[self.weightsSD],
                    scale_diag=[1])

        self.allBiases=[None]*outputDims #List to store all bias distributions
        
        #bias mean value and mean distribution
        self.biasesMean=tf.Variable(0, dtype=self.dtype)
        self.biasesMeanHyper=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[2.0])

        #bias SD value and SD distribution
        self.biasesSD=tf.Variable(1, dtype=self.dtype)
        self.biasesSDHyper=tfd.MultivariateNormalDiag(loc=[self.weightsSD],
                    scale_diag=[1])
        
        #Starting weight mean, weight SD, bias mean, and bias SD
        self.firstHypers=np.float32([0, 1, 0, 1])
        #Create the weight and bias distributions for the perceptrons in a layer
        for n in range(self.outputDims):
            self.allWeights[n], self.allBiases[n] = self.createPerceptron()
            
        #Placeholders for the weights, biases, and their means and SDs for use in HMC
        self.weights_chain_start = tf.placeholder(dtype, shape=(self.outputDims, self.inputDims))
        self.bias_chain_start = tf.placeholder(dtype, shape=(self.outputDims, 1))
        self.chains=[self.weights_chain_start, self.bias_chain_start]
        
        self.weight_mean_chain_start = tf.placeholder(dtype, shape=())
        self.weight_SD_chain_start = tf.placeholder(dtype, shape=())
        
        self.bias_mean_chain_start = tf.placeholder(dtype, shape=())
        self.bias_SD_chain_start = tf.placeholder(dtype, shape=())
        self.hyper_chains=[self.weight_mean_chain_start,
                           self.weight_SD_chain_start,self.bias_mean_chain_start,
                           self.bias_SD_chain_start]
    
    def createPerceptron(self):
        """Creates the weight and biases distributions for a single perceptron
        
        Returns:
            * weights: distribution for the perceptron's weights
            * bias: distribution for the perceptron's biases
        """
        weights = tfd.MultivariateNormalDiag(
                loc=np.ones(self.inputDims, dtype=self.dtype)*self.weightsMean,
                scale_diag=np.ones(self.inputDims, dtype=self.dtype)*self.weightsSD)

        bias = tfd.MultivariateNormalDiag(
                loc=[self.biasesMean],
                scale_diag=[self.biasesSD])
        
        return(weights, bias)
        
    def calculateProbs(self, weightBias):
        """Calculates the log probability of a set of weights and biases given
        their distributions in this layer.
        
        Arguments:
            * weights: new possible weight tensor
            * biases: new possible bias tensor
            
        Returns:
            * prob: log prob of weights and biases given their distributions
        """
        weights=weightBias[0]
        biases=weightBias[1]
        prob=0
        for m in range(self.outputDims):

            val=self.allWeights[m].log_prob([weights[m]])
            prob+=tf.reduce_sum(val)
            val=self.allBiases[m].log_prob([biases[m]])
            prob+=tf.reduce_sum(val)
        return(prob)
    def calculateHyperProbs(self, hypers, weightBias):
        """Calculates the log probability of a set of weights and biases given
        new distribtuions as well as the probability of the new distribution
        means and SDs given their distribtuions.
        
        Arguments:
            * weights: current weight tensor
            * weightsMean: new possible weight mean
            * weightsSD: new possible weigh SD
            * biases: current bias tensor
            * biasesMean: new possible bias mean
            * biasesSD: new possible bias SD
            
        Returns:
            * prob: log probability of weights and biases given the new distributions 
            and the probability of the new distributions given their priors
        """
        weightsMean=hypers[0]
        weightsSD=hypers[1]
        biasesMean=hypers[2]
        biasesSD=hypers[3]
        weights=weightBias[0]
        biases=weightBias[1]
        prob=0

        val=self.weightsMeanHyper.log_prob([[weightsMean]])
        prob+=tf.reduce_sum(val)
        val=self.weightsSDHyper.log_prob([[weightsSD]])
        prob+=tf.reduce_sum(val)

        val=self.biasesMeanHyper.log_prob([[biasesMean]])
        prob+=tf.reduce_sum(val)
        val=self.biasesSDHyper.log_prob([[biasesSD]])
        prob+=tf.reduce_sum(val)

        weightsDist = tfd.MultivariateNormalDiag(
                loc=np.ones(self.inputDims, dtype=self.dtype)*weightsMean,
                scale_diag=np.ones(self.inputDims, dtype=self.dtype)*weightsSD)

        biasDist = tfd.MultivariateNormalDiag(
                loc=[biasesMean],
                scale_diag=[biasesSD])
        
        val=weightsDist.log_prob([weights])
        prob+=tf.reduce_sum(val)
        val=biasDist.log_prob([biases])
        prob+=tf.reduce_sum(val)

        return(prob)
        
    def sample(self):
        """Creates randomized weight and bias tensors based off of their distributions
        
        Returns:
            * tempWeights: randomized weight tensor
            * tempBiases: randomized bias tensor
        """
        
        tempWeights = tf.Variable([self.allWeights[n].sample() for n in range(self.outputDims)])
        tempBiases = tf.Variable([self.allBiases[n].sample() for n in range(self.outputDims)])
        
        return(tempWeights, tempBiases)

    def update(self, hypers):
        """Updates the weight and bias distributions for the layer when their
        means and SDs change.
        
        Arguments:
            * weightsMean: mean of weight distribution
            * weightsSD: standard deviation of weight distribution
            * biasesMean: mean of bias distribution
            * biasesSD: standard deviation of bias distribution
        """
        #tf.assign(self.weightsMean,tf.Variable(weightsMean, dtype=self.dtype))
        #tf.assign(self.weightsSD,tf.Variable(weightsSD, dtype=self.dtype))
        #tf.assign(self.biasesMean,tf.Variable(biasesMean, dtype=self.dtype))
        #tf.assign(self.biasesSD,tf.Variable(biasesSD, dtype=self.dtype))
        self.weightsMean=tf.Variable(hypers[0], dtype=self.dtype)
        self.weightsSD=tf.Variable(hypers[1], dtype=self.dtype)
        self.biasesMean=tf.Variable(hypers[2], dtype=self.dtype)
        self.biasesSD=tf.Variable(hypers[3], dtype=self.dtype)
        for n in range(self.outputDims):
            self.allWeights[n], self.allBiases[n] = self.createPerceptron()
            #tf.assign(self.allWeights[n].loc,tf.Variable(np.ones(self.inputDims, dtype=self.dtype)*weightsMean))
            #tf.assign(self.allBiases[n].scale,tf.diag(np.ones(self.inputDims, dtype=self.dtype)*weightsSD))
    
    def expand(self, current):
        currentShape=tf.pad(
                tf.shape(current),
                paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
                constant_values=1)
        return(tf.reshape(current, currentShape))
    
    def predict(self,inputTensor, weightBias):
        weightTensor=self.expand(weightBias[0])
        biasTensor=self.expand(weightBias[1])
        result=tf.add(tf.matmul(weightTensor, inputTensor), biasTensor)
        return(result)
