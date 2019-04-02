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
    def __init__(self, alpha=0.2):
        self.alpha=alpha
    def predict(self, inputTensor, _):
        result=nn.leaky_relu(inputTensor, self.alpha)
        return(result)
