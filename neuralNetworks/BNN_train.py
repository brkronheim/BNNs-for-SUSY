import click
import os

import numpy as np
import tensorflow as tf

from math import ceil
from BNN_functions import (normalizeData, build_input_pipeline, createNeuralNet, 
                           percentError, setupOptimization)

@click.command()
@click.option('--hidden', default=3, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--epochs', default=60, help='Number of epochs to train for')
@click.option('--tb', default=None, help='Folder for Tensorboard')
@click.option('--name', default=None, help='Name of network')

def main(hidden, width, epochs, tb, name):
    """This script creates a Bayesian Neural Network using Dense Flipout Layers from 
    TensorFlow-Probability. Currently the network is trained using however many 
    hidden layers the user specifies with the depth specified and trained for the
    number of epochs specified. The hidden layers use a PRELU activation.
    
    The optimizer used is the Adam Optimizer with a learning rate of 0.001 and an
    epsilon of 1E-08.
    
    This script will connect to Tensorboard and create plots of validation error
    and validation percent difference vs. epoch if a folder for it is input by
    the user in tb. 
    
    Additionally, if the user gives the network a name in --name this program 
    will save the network using that name.
    """
    
    #Load training and validation data
    trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
    trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
    valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
    valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)


    #Normalize the training and output data and collect the values used to do so
    normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 
    
    graph1=tf.Graph()
    with graph1.as_default():
        
        #Create the iterators used for training and validation
        #Path for tensorboard to save data
        if(tb is not None):
            STORE_PATH = os.path.join(os.getcwd(),tb)
        
        #hyper paramaters
        batch_size=128 
        learning_rate=0.001
        
        #dropout paramaters
        dropoutPercent=0.0
        rate=tf.placeholder(dtype=tf.float32, shape=(), name="rate")
        
        #size of data
        train_size=len(trainIn[:,1])
        val_size=len(valIn[:,1])
        data_size=train_size+val_size
        
        #setup data pipelines
        (x_input, y_output, handle, training_iterator, validation_iterator) = build_input_pipeline(
           data, batch_size)

        #Create the neural network
        neural_net, logits = createNeuralNet(width, hidden, x_input, rate)

        #Print a network summary
        neural_net.summary()

        #Create the percent difference metric
        percentErr = percentError(normInfo[0][0], normInfo[0][1], y_output, logits)

        #Create the loss function and optimizer
        loss, train_op = setupOptimization(normInfo[0][0], normInfo[0][1], learning_rate, y_output, logits)


        init_op= tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

        #merge outputs for tensorboard
        if(tb is not None):
            merged = tf.summary.merge_all()

        with tf.Session(graph=graph1) as sess:
            if(tb is not None):
                writer = tf.summary.FileWriter(STORE_PATH, sess.graph) #Tensorboard writer
            
            sess.run(init_op)

            train_handle = sess.run(training_iterator.string_handle())
            validate_handle = sess.run(validation_iterator.string_handle())
            
            steps=ceil(train_size/batch_size) #Number of batches to get through all the data

            for j in range(epochs):
                averageLoss=0
                averageError=0

                #Run the training cycle
                for i in range(steps):
                    loss_value, error_value, _ = sess.run([loss, percentErr, train_op],
                               feed_dict={handle: train_handle, rate: dropoutPercent})

                    averageLoss+=loss_value
                    averageError+=error_value

                print("Epoch: {:>3d} Training loss: {:.5f} Training Error: {:.3f}".format(
                    j+1, averageLoss/steps, averageError/steps))

                #Run the validation cycle
                valid_iters=1 #Numer of runs through the validation data. Note:
                              #adjusting this value will scale the output to 
                              #Tensorboard by the same amount

                averageLoss=0
                averageError=0
                if(tb is not None): #when writing to tensorboard
                    for i in range(valid_iters):
                        loss_value, error_value, summary = sess.run([loss, percentErr, merged],
                                             feed_dict={handle: validate_handle, rate: 0.0})
                        averageLoss+=loss_value
                        averageError+=error_value

                        writer.add_summary(summary, j+1)
                else: #when not writing to tensorboard
                    for i in range(valid_iters):
                        loss_value, error_value = sess.run([loss, percentErr],
                                             feed_dict={handle: validate_handle, rate: 0.0})
                        averageLoss+=loss_value
                        averageError+=error_value
                        
                print("Validation loss: {:.5f} Validation Percent Error: {:.3f} Iterations: {}".format(
                        averageLoss/valid_iters, averageError/valid_iters, valid_iters))
               
            #save the network
            if(name is not None):
                saver = tf.train.Saver()
                print('\nSaving...')
                saver.save(sess, "./"+name)
            

if(__name__=="__main__"):
    main()
