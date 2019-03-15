# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:45:59 2019

@author: brade
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.ops import gen_nn_ops

import layer

tfd = tfp.distributions

class network(object):
    def __init__(self, iters, inputDims, hiddenDims, hiddenLayers, outputDims, 
                 dtype, x, y):

        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.hiddenLayers = hiddenLayers
        self.outputDims = outputDims
        self.dtype = dtype
        self.iters=iters

        self.x = tf.reshape(tf.constant(x, dtype=self.dtype),[len(x),1])
        self.y = tf.constant(y, dtype=self.dtype)        

        self.states=[]
        self.priors=[]
        self.vars_=[None]*(iters+1)

        self.currentNetworkVars=[]

        self.hyperStates=[]
        self.hyperPriors=[]
        self.hyperVars=[None]*(iters+1)

        self.layers=[]
        self.sampleVars=[]
        self.sampleHyperVars=[]
        
        self.makeLayer(inputDims, hiddenDims)
        
        for n in range(hiddenLayers-1):
            self.makeLayer(hiddenDims, hiddenDims)
        
        self.makeLayer(hiddenDims, outputDims)        
        for n in range(iters):
            self.vars_[n+1]=self.sampleVars
            self.hyperVars[n+1]=self.sampleHyperVars
        self.createInitialChains()
    def make_response_likelihood(self, *argv):
        """Make a prediction"""
        argv=argv[0]
        expandedStates=[None]*len(argv)
        for n in range(len(argv)):
            current=argv[n]
            currentShape=tf.pad(
                tf.shape(current),
                paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
                constant_values=1)
            expandedStates[n]=tf.reshape(current, currentShape)
        
        
        current=tf.transpose(self.x)
        n=0
        while(n+1<len(expandedStates)-2):
            current=gen_nn_ops.relu(tf.add(tf.matmul(expandedStates[n], current), expandedStates[n+1]))
            n+=2
        current=tf.add(tf.matmul(expandedStates[-2], current), expandedStates[-1])
        
        return tfd.Normal(
            loc=current,
            scale=np.array(.1, current.dtype.as_numpy_dtype))
        
    def metrics(self, mean, sd, *argv):
        """Make a prediction"""
        argv=argv[0]
        expandedStates=[None]*len(argv)
        for n in range(len(argv)):
            current=argv[n]
            currentShape=tf.pad(
                tf.shape(current),
                paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
                constant_values=1)
            expandedStates[n]=tf.reshape(current, currentShape)
        
        
        current=tf.transpose(self.x)
        n=0
        while(n+1<len(expandedStates)-2):

            current=gen_nn_ops.relu(tf.add(tf.matmul(expandedStates[n], current), expandedStates[n+1]))
            n+=2
        current=tf.add(tf.matmul(expandedStates[-2], current), expandedStates[-1])
        squaredError=tf.reduce_mean(tf.squared_difference(current, self.y))
        current=tf.add(tf.multiply(current, sd), mean)
        real=tf.add(tf.multiply(self.y, sd), mean)
        percentError=tf.reduce_mean(tf.multiply(tf.abs(tf.divide(tf.subtract(current, real), real)), 100))
        return(squaredError, percentError)
        
        
    def calculateProbs(self, *argv):
        prob=tf.reduce_sum(self.make_response_likelihood(argv).log_prob(self.y))
        
        for n in range(len(self.priors)):
            for m in range(len(self.priors[n])):     
                val=self.priors[n][m].log_prob([argv[n][m]])
                prob+=tf.reduce_sum(val)        
        
        prob=tf.reduce_sum(prob)
 
        return(prob)

    def calculateHyperProbs(self, *argv):
        prob=0
        for n in range(len(self.priors)):
            dist=None
            if(n%2==0):
                val=self.layers[int(n/2)].weightsMeanHyperMean.log_prob([[argv[2*n]]])
                prob+=tf.reduce_sum(val)
                val=self.layers[int(n/2)].weightsMeanHyperSD.log_prob([[argv[2*n+1]]])
                prob+=tf.reduce_sum(val)
                if(n<=1):                
                    dist=tfd.MultivariateNormalDiag(
                            loc=np.ones(self.inputDims, dtype=self.dtype)*argv[2*n],
                            scale_diag=np.ones(self.inputDims, dtype=self.dtype)*argv[2*n+1])
                else:                
                    dist=tfd.MultivariateNormalDiag(
                            loc=np.ones(self.hiddenDims, dtype=self.dtype)*argv[2*n],
                            scale_diag=np.ones(self.hiddenDims, dtype=self.dtype)*argv[2*n+1])
            else:
                val=self.layers[int(n/2)].biasesMeanHyperMean.log_prob([[argv[2*n]]])
                prob+=tf.reduce_sum(val)
                val=self.layers[int(n/2)].biasesMeanHyperSD.log_prob([[argv[2*n+1]]])
                prob+=tf.reduce_sum(val)
                dist=tfd.MultivariateNormalDiag(
                        loc=[argv[2*n]],
                        scale_diag=[argv[2*n+1]]) 
            for m in range(len(self.priors[n])):     
                val=dist.log_prob([self.currentNetworkVars[n][m]])
                prob+=tf.reduce_sum(val)        
        
        prob=tf.reduce_sum(prob)
        return(prob)

    
    def predict(self, *argv):
        argv=argv[0]
        expandedStates=[None]*len(argv)
        for n in range(len(argv)):
            current=argv[n]
            currentShape=tf.pad(
                tf.shape(current),
                paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
                constant_values=1)
            expandedStates[n]=tf.reshape(current, currentShape)
        
        
        current=tf.transpose(self.x)
        n=0
        while(n+1<len(expandedStates)-2):
            current=gen_nn_ops.relu(tf.add(tf.matmul(expandedStates[n], current), expandedStates[n+1]))
            n+=2
        current=tf.add(tf.matmul(expandedStates[-2], current), expandedStates[-1])
        return(current)
    
    def createInitialChains(self):
        firstVars=[]
        firstHyperVars=[]
        tempWeights0 = tf.random_normal([self.hiddenDims, self.inputDims], dtype =self.dtype)
        tempBiases0 = tf.random_normal([self.hiddenDims, 1], dtype  =self.dtype)
        
        firstVars.append(tempWeights0)
        firstVars.append(tempBiases0)
        
        for n in range(self.hiddenLayers-1):
            tempWeights0 = tf.random_normal([self.hiddenDims, self.hiddenDims], dtype=self.dtype)
            tempBiases0 = tf.random_normal([self.hiddenDims, 1], dtype=self.dtype)
            
            firstVars.append(tempWeights0)
            firstVars.append(tempBiases0)

        tempWeights0 = tf.random_normal([self.outputDims, self.hiddenDims], dtype=self.dtype)
        tempBiases0 = tf.random_normal([self.outputDims, 1], dtype =self.dtype)
        
        firstVars.append(tempWeights0)
        firstVars.append(tempBiases0)

        self.vars_[0]=firstVars
        self.currentNetworkVars=firstVars
        for layers in self.layers:
            for val in layers.firstHypers:    
                firstHyperVars.append(val)
        self.hyperVars[0]=firstHyperVars        
    
    
    def newPrediction(self):
        newState=[None]*len(self.priors)
        for n in range(len(self.priors)):
            newState[n] = [prior.sample() for prior in self.priors[n]]
        return(self.predict(newState))
        
    def makeLayer(self, inputDims, outputDims):

        tempLayer=layer.layer(inputDims, outputDims, self.dtype)
        
        self.layers.append(tempLayer)
        
        self.states.append(tempLayer.weights_chain_start)
        self.states.append(tempLayer.bias_chain_start)
        
        self.priors.append(tempLayer.allWeights)
        self.priors.append(tempLayer.allBiases)
        
        
        self.sampleVars.append(tf.zeros([outputDims, inputDims], self.dtype))
        self.sampleVars.append(tf.zeros([outputDims, 1], self.dtype))
        
        self.hyperStates.append(tempLayer.weight_mean_chain_start)
        self.hyperStates.append(tempLayer.weight_SD_chain_start)
        
        self.hyperStates.append(tempLayer.bias_mean_chain_start)
        self.hyperStates.append(tempLayer.bias_SD_chain_start)
        
        self.sampleHyperVars.append(tf.zeros([1], self.dtype))
        self.sampleHyperVars.append(tf.zeros([1], self.dtype))
        self.sampleHyperVars.append(tf.zeros([1], self.dtype))
        self.sampleHyperVars.append(tf.zeros([1], self.dtype))
        
    def update(self, newVals):
        newPriors=[]
        for n in range(len(self.priors)):
            entry=[]
            dist=None
            if((n/2)%2==0):
                if(n<=1):                
                    dist=tfd.MultivariateNormalDiag(
                            loc=tf.Variable(np.ones(self.inputDims, dtype=self.dtype))*newVals[2*n],
                            scale_diag=np.ones(self.inputDims, dtype=self.dtype)*newVals[2*n+1])
                else:                
                    dist=tfd.MultivariateNormalDiag(
                            loc=tf.Variable(np.ones(self.hiddenDims, dtype=self.dtype))*newVals[2*n],
                            scale_diag=np.ones(self.hiddenDims, dtype=self.dtype)*newVals[2*n+1])
            else:
               dist=tfd.MultivariateNormalDiag(
                        loc=[newVals[2*n]],
                        scale_diag=[newVals[2*n+1]]) 
            
            for m in range(len(self.priors[n])):     
                entry.append(dist)
            newPriors.append(entry)
        self.priors=newPriors
        
