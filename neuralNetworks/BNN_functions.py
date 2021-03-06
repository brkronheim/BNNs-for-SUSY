import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scipy import stats

def normalizeData(trainIn, trainOut, valIn, valOut):
    """Normalizes the training and validation data to improve network training.
    
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
    
    trainOutput-=mean
    trainOutput/=sd
    valOutput-=mean
    valOutput/=sd
    
    
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
    
def build_input_pipeline(data, train_batch, randomize=True, val_batch = None, handle = None):
    """Build an Iterator which can be switched between training and validation data.
    
    The Iterator here allows for data to be fed in very rapidly during the 
    training and validation phases.
    
    Input:
        * data: An array containing [trainIn,trainOutput,valIn,valOutput]
        * train_batch: An integer corresponding to the batch size for the training iterator
        * randomize: A boolean saying whether to shuffle the training data each iteration
        * val_Batch: Batch size for validation
        * handle: Option to pass in previously created handle
    
    Returns:
        * x_input: Symbolic representation of the input data the iterator produces
        * y_output: Symbolic representation of the output data the iterator produces
        * handle: A handle for the iterator
        * training_iterator: A version of the iterator over the training data
        * validation_iterator: A versionof the iterator over the validation data
        
    """

    # Build an iterator over training batches.
    # Data will be shuffled every time it is iterated through if randomize = True
    # Shuffling uses a buffer of size 50000
    
    training_dataset = tf.data.Dataset.from_tensor_slices(
                    (data[0].astype(np.float32), data[1].astype(np.float32)))
    if(randomize):
        training_dataset = training_dataset.shuffle(50000, reshuffle_each_iteration=True)
    training_dataset = training_dataset.repeat().batch(train_batch)
    
    training_iterator = training_dataset.make_one_shot_iterator()
    
    # Build a iterator over the validation set with batch_size=val_Batch,
    if(val_batch is None):
        val_batch=len(data[3])
    
    validation_dataset = tf.data.Dataset.from_tensor_slices((data[2].astype(np.float32), data[3].astype(np.float32)))
    validation_frozen = validation_dataset.repeat().batch(val_batch)
    validation_iterator = validation_frozen.make_one_shot_iterator()
    
    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    if(handle is None):
        handle = tf.placeholder(tf.string, shape=[], name = "handle")

    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    
    x_input, y_output = feedable_iterator.get_next() #Represent the output from the iterator

    return x_input, y_output, handle, training_iterator, validation_iterator

def createNeuralNet(layer_width, hidden_layers, x_input, rate):
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
    
    with tf.name_scope("bayesian_neural_net", values=[x_input, rate]):
        neural_net = tf.keras.Sequential()
        for n in range(hidden_layers):
            """
            neural_net.add(tfp.layers.DenseLocalReparameterization(layer_width, activation=tf.nn.relu,
                               bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                               bias_prior_fn=tfp.layers.default_multivariate_normal_fn))
            """
            neural_net.add(tf.keras.layers.Dense(layer_width, activation=None))
            
            neural_net.add(tf.keras.layers.PReLU())
        neural_net.add(tf.keras.layers.Dense(layer_width, activation=None))
        """    
        neural_net.add(tfp.layers.DenseLocalReparameterization(1, activation=None,
                               bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                               bias_prior_fn=tfp.layers.default_multivariate_normal_fn))
        """
        logits = tf.identity(neural_net(x_input), name="logits")
        
        return(neural_net,logits)

def percentError(mean, sd, y_output, logits):
    """ Calculates the percent error between the prediction and real value. 
    
    The percent error is calculated with the formula:
        100*(|real - predicted|)/(real) 
    The real and predicted values are un normalized to see how accurate the true
    predictions are. This metric is created in the name scope "percentError".
    
    Input:
        * mean: The mean of the original output distribution
        * sd: The standard deviation of the original output distribution
        * y_output: The y_output symbolic output from the iterator
        * logits: The symbolic prediction output from the nerual network
    
    Returns:
        * percentErr: An operation which calculates the percent error when
        *              used in a training or validation run of the network
    
    """
    
    with tf.name_scope("percentError", values=[y_output, logits]):
        predictions= tf.exp(tf.reduce_sum(logits, axis=-1)*sd + mean) 
        actualValue = tf.exp(y_output*sd + mean)
        
        percentErr = tf.reduce_mean(abs((actualValue-predictions)*100/(actualValue)))
        
        tf.summary.scalar('Percent_Error', percentErr)
        
        return(percentErr)
    
def setupOptimization(mean, sd, learning_rate, y_output, logits):
    """Sets up the optimizer and the loss function 
    
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
        labels=y_output#*sd+mean
        #labels=tf.exp(labels)
        predictions=tf.reduce_sum(logits,axis=-1)#*sd+mean
        #predictions=tf.exp(predictions)
        loss=tf.losses.mean_squared_error(labels=labels, predictions=predictions)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        
        tf.summary.scalar('Mean_Squared_Error', loss)
        
        return(loss, train_op)
    
def runInference(logits):
    """Operation which returns the predicted value of a given input.
       
       Input:
           * logits: predicted value
       OutputL
           * logits: usable predicted value
    """
    with tf.name_scope("runInference", values=[logits]):
        return(tf.reduce_sum(logits, axis=-1))
