import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.ops import gen_nn_ops

tfd = tfp.distributions

class network(object):
    """ An object used for storing all of the variables required to create
    a Bayesian Neural Network using Hamiltonian Monte Carlo and then training
    the network.
    """
    def __init__(self, inputDims, hiddenDims, hiddenLayers, outputDims, 
                 dtype, trainX, trainY, validateX, validateY):
        """
        Arguments:
            
        
        """
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.hiddenLayers = hiddenLayers
        self.outputDims = outputDims
        self.dtype = dtype

        self.trainX = tf.reshape(tf.constant(trainX, dtype=self.dtype),[len(trainX),1])
        self.trainY = tf.constant(trainY, dtype=self.dtype)        

        self.validateX = tf.reshape(tf.constant(validateX, dtype=self.dtype),[len(validateX),1])
        self.validateY = tf.constant(validateY, dtype=self.dtype)        

        self.vars_=[]
        self.hyperVars=[]

        self.states=[]
        self.hyperStates=[]

        self.layers=[]

                
        
    
    def make_response_likelihood(self, *argv):
        """Make a prediction and assign it a distribution
        
        Arguments:
            *argv:an undetermined number of tensors containg the weights
            and biases
        Returns:
            *result: a normal distribution centered about the prediction with
            a standard deviation of 0.1
        """
        current=self.predict(True, argv[0])
        #Prediction distribution
        result=tfd.Normal(
            loc=current,
            scale=np.array(.1, current.dtype.as_numpy_dtype))
        return(result) 
        
    def metrics(self, train, mean=1, sd=1, *argv):
        """Calculates the average squared error and percent difference of the 
        current network
        Arguments:
            * mean: mean value used for unshifiting a distribution
            * sd: sd value used for unscalling a distribution
            * argv: an undetermined number of tensors containg the weights
            and biases
        Returns:
            * squaredError: the mean squared error of predictions from the network
            * percentError: the percent error of the predictions from the network
        
        """
        tensors=argv
        if(len(tensors)==0):
            tensors=self.vars_
        else:
            tensors=tensors[0]
        
        
        current=self.predict(train,tensors)
        y=self.validateY
        if(train):
            y=self.trainY
        
        squaredError=tf.reduce_mean(tf.squared_difference(current, y))
        current=tf.add(tf.multiply(current, sd), mean)
        real=tf.add(tf.multiply(y, sd), mean)
        percentError=tf.reduce_mean(tf.multiply(tf.abs(tf.divide(tf.subtract(current, real), real)), 100))
        return(squaredError, percentError)
        
        
    def calculateProbs(self, *argv):
        """Calculates the log probability of the current weight and bias values
        as well as the log probability of their prediction.
        
        """

        prob=tf.reduce_sum(self.make_response_likelihood(argv).log_prob(self.trainY))
        
        for n in range(len(self.layers)):
            prob+=self.layers[n].calculateProbs(argv[2*n], argv[2*n+1])
        return(prob)

    def calculateHyperProbs(self, *argv):
        prob=0
        for n in range(len(self.layers)):
            prob+=self.layers[n].calculateHyperProbs(self.vars_[2*n],
                             argv[4*n], argv[4*n+1],self.vars_[2*n+1],
                             argv[4*n+2],argv[4*n+3])
        return(prob)

    
    def predict(self, train, *argv):
        tensors=argv
        if(len(tensors)==0):
            tensors=self.vars_
        else:
            tensors=tensors[0]
        x=self.trainX
        if(not train):
            x=self.validateX
        expandedStates=[None]*len(tensors)
        for n in range(len(tensors)):
            current=tensors[n]
            currentShape=tf.pad(
                tf.shape(current),
                paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
                constant_values=1)
            expandedStates[n]=tf.reshape(current, currentShape)
        
        
        current=tf.transpose(x)
        n=0
        while(n+1<len(expandedStates)-2):
            current=gen_nn_ops.relu(tf.add(tf.matmul(expandedStates[n], current), expandedStates[n+1]))
            n+=2
        current=tf.add(tf.matmul(expandedStates[-2], current), expandedStates[-1])
        return(current)
    
    def createInitialChains(self):

        self.hyperVars=[]
        tempWeights0 = tf.random_normal([self.hiddenDims, self.inputDims], dtype =self.dtype)
        tempBiases0 = tf.random_normal([self.hiddenDims, 1], dtype  =self.dtype)
        
        self.vars_.append(tempWeights0)
        self.vars_.append(tempBiases0)
        
        for n in range(self.hiddenLayers-1):
            tempWeights0 = tf.random_normal([self.hiddenDims, self.hiddenDims], dtype=self.dtype)
            tempBiases0 = tf.random_normal([self.hiddenDims, 1], dtype=self.dtype)
            
            self.vars_.append(tempWeights0)
            self.vars_.append(tempBiases0)

        tempWeights0 = tf.random_normal([self.outputDims, self.hiddenDims], dtype=self.dtype)
        tempBiases0 = tf.random_normal([self.outputDims, 1], dtype =self.dtype)
        
        self.vars_.append(tempWeights0)
        self.vars_.append(tempBiases0)

        for layers in self.layers:
            for val in layers.firstHypers:    
                self.hyperVars.append(val)
   
    
    def update(self):
        for n in range(len(self.layers)):
            self.layers[n].update(self.hyperVars[4*n],self.hyperVars[4*n+1],
                       self.hyperVars[4*n+2],self.hyperVars[4*n+3])
        
    def add(self, layer):
        self.layers.append(layer)
        
        self.states.append(layer.weights_chain_start)
        self.states.append(layer.bias_chain_start)
        
        self.hyperStates.append(layer.weight_mean_chain_start)
        self.hyperStates.append(layer.weight_SD_chain_start)
        
        self.hyperStates.append(layer.bias_mean_chain_start)
        self.hyperStates.append(layer.bias_SD_chain_start)
        
        """
        tempWeights = tf.random_normal([layer.inputDims, layer.outputDims], dtype =self.dtype)
        tempBiases= tf.random_normal([self.outputDims, 1], dtype  =self.dtype)
        
        self.vars_.append(tempWeights)
        self.vars_.append(tempBiases)
        
        for val in layer.firstHypers:    
            self.hyperVars.append(val)
        """
        
    def setup(self):
        self.createInitialChains()
        self.setupMCMC()
        
    def setupMCMC(self):
        self.step_size = tf.Variable(np.array(0.0001, self.dtype))
        self.hyper_step_size = tf.Variable(np.array(0.001, self.dtype))
        num_results = 2 #number of markov chain draws
        
        self.states_MCMC, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0, #start collecting data on first step
            current_state=self.states, #starting parts of chain 
            kernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
                target_log_prob_fn=self.calculateProbs, #used to calculate log density
                num_leapfrog_steps=20,
                step_size=self.step_size,
                
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(decrement_multiplier=0.01), 
                state_gradients_are_stopped=True))
        self.avg_acceptance_ratio = tf.reduce_mean(
            tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)))
        self.loss = -tf.reduce_mean(kernel_results.accepted_results.target_log_prob)
        
        
        self.hyper_states_MCMC, hyper_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0, #start collecting data on first step
            current_state=self.hyperStates, #starting parts of chain 
            kernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
                target_log_prob_fn=self.calculateHyperProbs, #used to calculate log density
                num_leapfrog_steps=20,
                step_size=self.hyper_step_size,
                
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(decrement_multiplier=0.01), 
                state_gradients_are_stopped=True))
        self.hyper_avg_acceptance_ratio = tf.reduce_mean(
            tf.exp(tf.minimum(hyper_kernel_results.log_accept_ratio, 0.)))
        self.hyper_loss = -tf.reduce_mean(hyper_kernel_results.accepted_results.target_log_prob)
        
    def train(self, epochs, updatePriorsWait, updatePriorsEpochs, startSampling, samplingStep):
        self.results=[]
        with tf.Session() as sess:
        
            init_op = tf.global_variables_initializer()
            init_op.run()
        
            iter_=0
            
            while(iter_<epochs):
                for n in range(len(self.vars_)):
                    if(tf.contrib.framework.is_tensor(self.vars_[n])):
                        self.vars_[n]=sess.run(self.vars_[n])
                [
                    nextStates,
                    loss_,
                    step_size_,
                    avg_acceptance_ratio_,
                ] = sess.run([
                    self.states_MCMC,
                    self.loss,
                    self.step_size,
                    self.avg_acceptance_ratio,
                ], feed_dict={tuple(self.states): tuple(self.vars_)})
                for n in range(len(self.vars_)):
                    self.vars_[n]=nextStates[n][-1]
                iter_+=1
                print('iter:{:>2}  Network loss:{: 9.3f}  step_size:{:.7f}  avg_acceptance_ratio:{:.4f}'.format(
                          iter_, loss_, step_size_, avg_acceptance_ratio_))
                if(iter_%updatePriorsWait==0):
                    hyperIter=0
                    while(hyperIter<updatePriorsEpochs):
                        for n in range(len(self.hyperVars)):
                            if(tf.contrib.framework.is_tensor(self.hyperVars[n])):
                                self.hyperVars[n]=sess.run(self.hyperVars[n])
                        [
                            nextHyperStates,
                            hyperLoss_,
                            hyper_step_size_,
                            hyper_avg_acceptance_ratio_,
                        ] = sess.run([
                            self.hyper_states_MCMC,
                            self.hyper_loss,
                            self.hyper_step_size,
                            self.hyper_avg_acceptance_ratio,
                        ], feed_dict={tuple(self.hyperStates): tuple(self.hyperVars)})
                        for n in range(len(self.hyperVars)):
                            self.hyperVars[n]=nextHyperStates[n][-1]
                        hyperIter+=1
                        print('iter:{:>2}  Hyper loss:{: 9.3f}  step_size:{:.7f}  avg_acceptance_ratio:{:.4f}'.format(
                                hyperIter, hyperLoss_, hyper_step_size_, hyper_avg_acceptance_ratio_))
                        if(hyperIter==updatePriorsWait):
                            self.update()
                
                if(iter_>=startSampling and iter_%samplingStep==0):
                    squaredError, percentError=sess.run(self.metrics(False))
                    print('squaredError{: 9.5f} percentDifference{: 7.3f}'.format(squaredError, percentError))
                    out=self.predict(False)
                    self.results.append(sess.run(out))
        return(self.results)
