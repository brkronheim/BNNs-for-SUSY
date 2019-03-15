# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:57:29 2019

@author: brade
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class layer(object):
    class perceptron(object):
        """Stores the weight and biases distributions for a single perceptron"""
        def __init__(self, wMean, wSD, bMean, bSD, inputSize, dtype):
            """
            Arguments:
                * wMean: mean for weight distribution
                * wSD: standard deviation for weight distribution
                * bMean: mean for bias distribution
                * bSD: standard deviation for bias distribution
            """
            self.weights = tfd.MultivariateNormalDiag(
                    loc=np.ones(inputSize, dtype=dtype)*wMean,
                    scale_diag=np.ones(inputSize, dtype=dtype)*wSD)
    
            self.bias = tfd.MultivariateNormalDiag(
                    loc=[bMean],
                    scale_diag=[bSD])

            
    def __init__(self, inputSize, outputSize, dtype):
        self.inputSize=inputSize
        self.outputSize=outputSize
        self.dtype=dtype
        
        self.allWeights=[None]*outputSize

        self.weightsMean=tf.Variable(0, dtype=self.dtype)
        self.weightsMeanHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        self.weightsMeanHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        
        self.weightsSD=tf.Variable(1, dtype=self.dtype)
        self.weightsSDHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsSD],
                    scale_diag=[1.0])
        self.weightsSDHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])

        self.allBiases=[None]*outputSize

        self.biasesMean=tf.Variable(0, dtype=self.dtype)
        self.biasesMeanHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        self.biasesMeanHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])

        self.biasesSD=tf.Variable(1, dtype=self.dtype)
        self.biasesSDHyperMean=tfd.MultivariateNormalDiag(loc=[self.weightsSD],
                    scale_diag=[1.0])
        self.biasesSDHyperSD=tfd.MultivariateNormalDiag(loc=[self.weightsMean],
                    scale_diag=[1.0])
        self.firstHypers=np.float32([0.0, 1.0, 0.0, 1.0])

        for n in range(self.outputSize):
            node=layer.perceptron(self.weightsMean, self.weightsSD, self.biasesMean, self.biasesSD, self.inputSize, self.dtype)
            self.allWeights[n]=node.weights
            self.allBiases[n]=node.bias
        
        self.weights_chain_start = tf.placeholder(dtype, shape=(self.outputSize, self.inputSize))
        self.bias_chain_start = tf.placeholder(dtype, shape=(self.outputSize, 1))
        
        self.weight_mean_chain_start = tf.placeholder(dtype, shape=())
        self.weight_SD_chain_start = tf.placeholder(dtype, shape=())
        
        self.bias_mean_chain_start = tf.placeholder(dtype, shape=())
        self.bias_SD_chain_start = tf.placeholder(dtype, shape=())
        
        
        
        
    def update(self, weightsMean, weightsSD, biasesMean, biasesSD):
        self.weightsMean=tf.Variable(weightsMean, dtype=self.dtype)
        self.weightsSD=tf.Variable(weightsSD, dtype=self.dtype)
        self.biasesMean=tf.Variable(biasesMean, dtype=self.dtype)
        self.biasesSD=tf.Variable(biasesSD, dtype=self.dtype)
        for n in range(self.outputSize):
            node=layer.perceptron(self.weightsMean, self.weightsSD, self.biasesMean, self.biasesSD, self.inputSize, self.dtype)
            self.allWeights[n]=node.weights
            self.allBiases[n]=node.bias
        
  
