"""BNN.py

This script creates a Bayesian Neural Network using Dense Flipout Layers from 
TensorFlow-Probability. Currently the network is trained using however many 
hidden layers the user specifies with the depth specified and trained for the
number of epochs specified. The hidden layers use a leaky RELU activation.

The optimizer used is the Adam Optimizer with a learning rate of 0.001 and an
epsilon of 1E-08.

This script will connect to Tensorboard and create plots of validation error
and validation percent difference vs. epoch. The folder for each run will be 
the one input by the user 

Usage = python3 BNN.py [Hidden] [Width] [Epochs] [Folder]
    Hidden  Number of hidden layers
    Width   Width of the hidden layers
    Epochs  Number of epochs to train for
    Folder  Folder for tensorboard information on this run
"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
import sys
import os
from scipy import stats
from math import ceil

def normalizeData(trainIn, trainOut, valIn, valOut):
    """ Normalizes the training and validation data to improve network training.
    
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
    trainOut-=mean
    trainOut/=sd
    valOut-=mean
    valOut/=sd
    
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
    
def build_input_pipeline(data, batch_size, validation_size):
    """Build an Iterator which can be switched between training and validation data.
    
    The Iterator here allows for data to be fed in very rapidly during the 
    training and validation phases.
    
    Input:
        * data: An array containing [trainIn,trainOutput,valIn,valOutput]
        * batch_size: An integer corresponding to the output size of a single
                      set of data from the training iterator
        * validation_size: The number of data entries in the validation data
    
    Returns:
        * x_input: Symbolic representation of the input data the iterator produces
        * y_output: Symbolic representation of the output data the iterator produces
        * handle: A handle for the iterator
        * training_iterator: A version of the iterator over the training data
        * validation_iterator: A versionof the iterator over the validation data
    """

    # Build an iterator over training batches.
    # Data will be shuffled every time it is iterated through
    # Suffling uses a buffer of size 50000
    training_dataset = tf.data.Dataset.from_tensor_slices(
                    (data[0].astype(np.float32), data[1].astype(np.float32)))
    training_batches = training_dataset.shuffle(
                    50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()
    
    # Build a iterator over the validation set with batch_size=validation_size,
    # This means a single batch will be the whole dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices((data[2].astype(np.float32), data[3].astype(np.float32)))
    validation_frozen = (validation_dataset.take(validation_size).
                    repeat().batch(validation_size))
    validation_iterator = validation_frozen.make_one_shot_iterator()
    
    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    
    x_input, y_output = feedable_iterator.get_next() #Represent the output from the iterator
    
    return x_input, y_output, handle, training_iterator, validation_iterator

def createNeuralNet(layer_width, hidden_layers, x_input):
    """ Creates a nerual net of dense flipout layers.
    
    This function creates a neural network using the specified number
    of hidden layers with a width corresponding to the specified layer
    width. It is a Bayesian Neural Network using the DenseFlipout layers.
    The output layer has no activation and the others use leaky_relu. The
    neural network is implemented through Keras Sequential as included in
    TensorFlows. The whole network is created in the name scope "bayesian_
    neural_net".
    
    Input:
        * layer_width: An integer corresponding to the width of the hidden layers 
        * hidden_layers: An integer corresponding to the number of the hidden layers
        * x_input: The x_input symbolic output from the iterator
        
    Returns:
        * nerual_net: A Keras Sequential Neural Network
        * logits: Representation of the output of the network
    
    """
    
    with tf.name_scope("bayesian_neural_net", values=[x_input]):
        neural_net = tf.keras.Sequential()

        for n in range(hidden_layers):
            neural_net.add(tfp.layers.DenseFlipout(layer_width, activation=tf.nn.leaky_relu))

        neural_net.add(tfp.layers.DenseFlipout(1, activation=None))

        logits = neural_net(x_input)
    
        return(neural_net,logits)

def percentDifference(mean, sd, y_output, logits):
    """ Calculates the percent difference of between the prediction and real value. 
    
    The percent difference is calculated with the formula:
        100*(|real - predicted|)/(real) 
    The real and predicted values are un normalized to see how accurate the true
    predictions are. This metric is created in the name scope "percentDifference".
    
    Input:
        * mean: The mean of the original output distribution
        * sd: The standard deviation of the original output distribution
        * y_output: The y_output symbolic output from the iterator
        * logits: The symbolic prediction output from the nerual network
    
    Returns:
        * percentDiff: An operation which calculates the percent difference when
        *              used in a training or validation run of the network
    
    """
    
    with tf.name_scope("percentDifference", values=[y_output, logits]):
        predictions= tf.exp(tf.reduce_sum(logits, axis=-1))*sd + mean
        actualValue = tf.exp(y_output)*sd + mean
        
        percentDiff = tf.reduce_mean(abs((actualValue-predictions)*100/(actualValue)))
        
        tf.summary.scalar('Percent_Difference', percentDiff)
        
        return(percentDiff)
    
def setupOptimization(learning_rate, y_output, logits):
    """ Sets up the optimizer and the loss function 
    
    The loss function used is the mean_squared_error and the optimizer
    is the Adam optimizer.
    
    Input:
        * learning_rate: A float corresponding to the learning rate for the 
        *                Adam optimizer
        * y_output: The y_output symbolic output from the iterator
        * logits: The symbolic prediction output from the nerual network
    
    Returns:
        * loss: An operation which calculates the mean squared error of a prediction
        * train_op: An operation which makes the nerual network undergo backprop 
    
    """
    
    with tf.name_scope("train", values=[y_output, logits]):
        loss=tf.losses.mean_squared_error(labels=y_output, predictions=tf.reduce_sum(logits,axis=-1))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        
        tf.summary.scalar('Mean_Squared_Error', loss)
        
        return(loss, train_op)

def main():
    #Access user inputs
    hidden_layers=int(sys.argv[1])
    layer_width=int(sys.argv[2])
    epochs = int(sys.argv[3])
    tensorboardFolder = (sys.argv[4])+"/"
    
    #Load training and validation data
    trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
    trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
    valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
    valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)

    #Path for tensorboard to save data
    STORE_PATH = os.path.join(os.getcwd(),tensorboardFolder)
    
    batch_size=128 
    learning_rate=0.001
    
    train_size=len(trainIn[:,1])
    val_size=len(trainOut[:,1])
    data_size=train_size+val_size

    #Normalize the training and output data and collect the values used to do so
    normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 
    
    #Create the iterators used for training and validation
    (x_input, y_output, handle, training_iterator, validation_iterator) = build_input_pipeline(
           data, batch_size, val_size)
    
    #Create the neural network
    neural_net, logits = createNeuralNet(layer_width, hidden_layers, x_input)
    
    #Print a network summary
    neural_net.summary()
    
    #Create the percent difference metric
    percentDiff = percentDifference(normInfo[0][0], normInfo[0][1], y_output, logits)
    
    #Create the loss function and optimizer
    loss, train_op = setupOptimization(learning_rate, y_output, logits)
    
    
    init_op= tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
    
    #merge outputs for tensorboard
    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(STORE_PATH, sess.graph) #Tensorboard writer
        
        sess.run(init_op)
        
        train_handle = sess.run(training_iterator.string_handle())
        validate_handle = sess.run(validation_iterator.string_handle())

        steps=ceil(train_size/batch_size) #Number of batches to get through all the data
        
        for j in range(epochs):
            averageLoss=0
            averageDiff=0

            #Run the training cycle
            for i in range(steps):
                loss_value, error_value, _ = sess.run([loss, percentDiff, train_op],
                           feed_dict={handle: train_handle})

                averageLoss+=loss_value
                averageDiff+=error_value
            
            print("Epoch: {:>3d} Training loss: {:.3f} Training Error: {:.3f}".format(
                j+1, averageLoss/steps, averageDiff/steps))
            
            #Run the validation cycle
            valid_iters=1 #Numer of runs through the validation data. Note:
                          #adjusting this value will scale the output to 
                          #Tensorboard by the same amount
            
            averageLoss=0
            averageDiff=0
            for i in range(valid_iters):
                loss_value, error_value, summary = sess.run([loss, percentDiff, merged],
                                     feed_dict={handle: validate_handle})
                averageLoss+=loss_value
                averageDiff+=error_value
                writer.add_summary(summary, j+1)
            
            print("Validation loss: {:.3f} Validation Error: {:.3f} Iterations: {}".format(
                    averageLoss/valid_iters, averageDiff/valid_iters, valid_iters))    
    
if(__name__=="__main__"):
    main()
