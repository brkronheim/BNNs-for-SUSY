import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from paramAdapter import paramAdapter

tfd = tfp.distributions

class network(object):
    """An object used for storing all of the variables required to create
    a Bayesian Neural Network using Hamiltonian Monte Carlo and then training
    the network.
    """
    def __init__(self, dtype, inputDims, trainX, trainY, validateX, validateY,mean,sd):
        """
        Arguments:
            * dtype: data type for Tensors
            * inputDims: dimension of input vector
            * trainX: the training data input, shape is n by inputDims
            * trainY: the training data output
            * validateX: the validation data input, shape is n by inputDims
            * validateY: the validation data output
            * mean: the mean used to scale trainY and validateY
            * sd: standard deviation used to scale trainY and validateY
        """
        self.dtype = dtype
        
        self.mean=mean
        self.sd=sd

        self.trainX = tf.reshape(tf.constant(trainX, dtype=self.dtype),[len(trainX),inputDims])
        self.trainY = tf.constant(trainY, dtype=self.dtype)        

        self.validateX = tf.reshape(tf.constant(validateX, dtype=self.dtype),[len(validateX),inputDims])
        self.validateY = tf.constant(validateY, dtype=self.dtype)        

        self.vars_=[] #List with the current weight and bias values
        self.hyperVars=[] #List with the current values of the hyper parameters

        self.states=[] #List with the weight and bias state placeholders
        self.hyperStates=[] # List with hyper parameter state placeholders

        self.layers=[] #List of all the layers


    def make_response_likelihood(self, *argv):
        """Make a prediction and assign it a distribution
        
        Arguments:
            *argv: an undetermined number of tensors containg the weights
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
        
    def metrics(self, logits, scaleExp, train, mean=1, sd=1):
        """Calculates the average squared error and percent difference of the 
        current network
        Arguments:
            * logits: output from the network
            * scaleExp: boolean value to determine whether to take the exponential
            of the data and scale it
            * train: boolean value to determine whether to use the training data 
            * mean: mean value used for unshifiting a distribution
            * sd: sd value used for unscalling a distribution
        Returns:
            * logits: output from the network
            * squaredError: the mean squared error of predictions from the network
            * percentError: the percent error of the predictions from the network
        
        """
        
        #Get the correct output values
        y=self.validateY
        if(train):
            y=self.trainY
        
        squaredError=tf.reduce_mean(tf.squared_difference(logits, y))
        scaled=tf.add(tf.multiply(logits, sd), mean)
        real=tf.add(tf.multiply(y, sd), mean)
        if(scaleExp):
            scaled=tf.exp(scaled)
            real=tf.exp(real)
        else:
            scaled=logits
        percentError=tf.reduce_mean(tf.multiply(tf.abs(tf.divide(tf.subtract(scaled, real), real)), 100))
        return(logits, squaredError, percentError)
        
        
    def calculateProbs(self, *argv):
        """Calculates the log probability of the current network values
        as well as the log probability of their prediction.
        
        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases.
        Returns:
            * prob: log probability of network values and network prediction
        """

        
        prob=tf.reduce_sum(self.make_response_likelihood(argv).log_prob(self.trainY))
        #probability of the network parameters
        index=0
        for n in range(len(self.layers)):
            numTensors=self.layers[n].numTensors
            if(numTensors>0):    
                prob+=self.layers[n].calculateProbs(self.vars_[index:index+numTensors])
                index+=numTensors
        return(prob)
        
        
    def calculateHyperProbs(self, *argv):
        """Calculates the log probability of the current hyper parameters
        
        Arguments:
            * argv: an undetermined number of tensors containg the hyper parameters
        Returns:
            * prob: log probability of hyper parameters given their priors
        """
        prob=0
        indexh=0
        index=0
        for n in range(len(self.layers)):
            numHyperTensors=self.layers[n].numHyperTensors
            numTensors=self.layers[n].numTensors
            if(numHyperTensors>0):    
                prob+=self.layers[n].calculateHyperProbs(argv[indexh:indexh+numHyperTensors],
                                 self.vars_[index:index+numTensors])
                indexh+=numHyperTensors
                index+=numTensors
        return(prob)

    
    def predict(self, train, *argv):
        """Makes a prediction
        
        Arguments:
            * train: a boolean value which determines whether to use training data
            * argv: an undetermined number of tensors containg the weights
            and biases.
            
        Returns:
            * prediction: a prediction from the network 
            
        """
        tensors=argv
        if(len(tensors)==0):
            tensors=self.vars_
        else:
            tensors=tensors[0]
        x=self.trainX
        if(not train):
            x=self.validateX

        prediction=tf.transpose(x)
        index=0
        for n in range(len(self.layers)):
            numTensors=self.layers[n].numTensors
            prediction=self.layers[n].predict(prediction,tensors[index:index+numTensors])
            index+=numTensors
        return(prediction)
    

    def add(self, layer, weights=None, biases=None):
        """Adds a new layer to the network
        Arguments:
            * layer: the layer to be added
            * weigths: matrix to initialize weights
            * biases: matrix to initialize biases
        """
        self.layers.append(layer)
        if(layer.numTensors>0):
            for states in layer.chains:
                self.states.append(states)
            if weights is None:
                startVals = layer.sample()
                for vals in startVals:
                    self.vars_.append(vals)
            else:
                self.vars_.append(tf.Variable(weights))
                self.vars_.append(tf.Variable(biases))

        if(layer.numHyperTensors>0):        
            for states in layer.hyper_chains:
                self.hyperStates.append(states)
            for vals in layer.firstHypers:
                self.hyperVars.append(vals)
            
            
    def setupMCMC(self, stepSize, stepMin, stepMax, leapfrog, leapMin, leapMax,
                  hyperStepSize, hyperLeapfrog, burnin, cores):
        """Sets up the MCMC algorithms
        Arguments:
            * stepSize: the starting step size for the weights and biases
            * stepMin: the minimum step size
            * stepMax: the maximum step size
            * leapfrog: number of leapfrog steps for weights and biases
            * leapMin: the minimum number of leapfrog steps
            * leapMax: the maximum number of leapfrog steps
            * hyperStepSize: the starting step size for the hyper parameters
            * hyperLeapfrog: leapfrog steps for hyper parameters
            * cores: number of cores to use
        Returns nothing
        """
        
        #Adapt the step size and number of leapfrog steps
        self.adapt=paramAdapter(stepSize,leapfrog,stepMin,stepMax,leapMin,leapMax,
                                2,burnin/2,4,0.1,cores)
        self.step_size = np.float32(stepSize)
        self.leapfrog = np.int64(leapfrog)
        self.step_size_placeholder=tf.placeholder(dtype=tf.float32,shape=())
        self.leapfrog_placeholder=tf.placeholder(dtype=tf.int64,shape=())
        
        
        self.hyper_step_size = tf.Variable(np.array(hyperStepSize, self.dtype))
        
        num_results = 1 #number of markov chain draws
        
        #Setup the Markov Chain for the network parameters
        self.states_MCMC, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0, #start collecting data on first step
            current_state=self.states, #starting parts of chain 
            parallel_iterations=cores, 
            kernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
                target_log_prob_fn=self.calculateProbs, #used to calculate log density
                num_leapfrog_steps=self.leapfrog_placeholder,
                step_size=self.step_size_placeholder,
                step_size_update_fn=None, 
                state_gradients_are_stopped=True,
                seed=100))
        self.avg_acceptance_ratio = tf.reduce_mean(
            tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)))
        self.loss = -tf.reduce_mean(kernel_results.accepted_results.target_log_prob)
        
        
        #Setup the Markov Chain for the hyper parameters
        self.hyper_states_MCMC, hyper_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0, #start collecting data on first step
            current_state=self.hyperStates, #starting parts of chain 
            parallel_iterations=cores,
            kernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
                target_log_prob_fn=self.calculateHyperProbs, #used to calculate log density
                num_leapfrog_steps=hyperLeapfrog,
                step_size=self.hyper_step_size,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(decrement_multiplier=0.01), 
                state_gradients_are_stopped=True,
                seed=50))
        self.hyper_avg_acceptance_ratio = tf.reduce_mean(
            tf.exp(tf.minimum(hyper_kernel_results.log_accept_ratio, 0.)))
        self.hyper_loss = -tf.reduce_mean(hyper_kernel_results.accepted_results.target_log_prob)
        
        
    def train(self, epochs, startSampling, samplingStep, mean=0, sd=1, scaleExp=False, folderName=None, 
              networksPerFile=1000, returnPredictions=False):
        """Trains the network
        Arguements:
            * Epochs: Number of training cycles
            * startSampling: Number of epochs before networks start being saved
            * samplingStep: Epochs between sampled networks
            * mean: true mean of output distribution
            * sd: true sd of output distribution
            * scaleExp: whether the metrics should be scaled via exp
            * folderName: name of folder for saved networks
            * networksPerFile: number of networks saved in a given file
            * returnPredictions: whether to return the prediction from the network
            
        Returns:
            * results: the output of the network when sampled (if returnPrediction=True)
        """
        
        #Create the folder and files for the networks
        filePath=None
        files=[]
        if(folderName is not None):
            filePath=os.path.join(os.getcwd(), folderName)
            if(not os.path.isdir(filePath)):
                os.mkdir(filePath)
            for n in range(len(self.vars_)):
                files.append(open(filePath+"/"+str(n)+".0"+".txt", "wb"))
            files.append(open(filePath+"/summary.txt", "wb"))
        
        if(returnPredictions):
            self.results=[]
        logits=self.predict(False, self.states) #prediction placeholder
        result, squaredError, percentError=self.metrics(logits, scaleExp, False, mean, sd)
        #get a prediction, squared error, and percent error
        
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            init_op.run()
            
            iter_=0
            while(iter_<epochs): #Main training loop
                #check that the vars are not tensors
                for n in range(len(self.vars_)):
                    if(tf.contrib.framework.is_tensor(self.vars_[n])):
                        self.vars_[n]=sess.run(self.vars_[n])
                for n in range(len(self.hyperVars)):
                            if(tf.contrib.framework.is_tensor(self.hyperVars[n])):
                                self.hyperVars[n]=sess.run(self.hyperVars[n])
                #Training step
                [
                    nextStates,
                    loss_,
                    avg_acceptance_ratio_,
                    result_, 
                    squaredError_, 
                    percentError_,
                    nextHyperStates,
                    hyperLoss_,
                    hyper_step_size_,
                    hyper_avg_acceptance_ratio_,
                ] = sess.run([
                    self.states_MCMC,
                    self.loss,
                    self.avg_acceptance_ratio,
                    result, 
                    squaredError, 
                    percentError,
                    self.hyper_states_MCMC,
                    self.hyper_loss,
                    self.hyper_step_size,
                    self.hyper_avg_acceptance_ratio,
                ], feed_dict={tuple(self.states+self.hyperStates+[self.leapfrog_placeholder,self.step_size_placeholder]):
                                tuple(self.vars_+self.hyperVars+[self.leapfrog,self.step_size])})
                
                
                for n in range(len(self.vars_)):
                    self.vars_[n]=nextStates[n][-1]
                for n in range(len(self.hyperVars)):
                    self.hyperVars[n]=nextHyperStates[n][-1]
                iter_+=1
                
                print('squaredError{: 9.5f} percentDifference{: 7.3f}'.format(squaredError_, percentError_))
                print()
                
                print('iter:{:>2}  Network loss:{: 9.3f}  step_size:{:.7f} leapfrog_num:{:>4} avg_acceptance_ratio:{:.4f}'.format(
                          iter_, loss_, self.step_size, self.leapfrog, avg_acceptance_ratio_))
                print('Hyper loss:{: 9.3f}  step_size:{:.7f}  avg_acceptance_ratio:{:.4f}'.format(
                                hyperLoss_, hyper_step_size_, hyper_avg_acceptance_ratio_))
                self.step_size, self.leapfrog=self.adapt.update(nextStates)
                
                #Create new files to record network
                if(iter_>startSampling and (iter_-1-startSampling)%(networksPerFile*samplingStep)==0):
                    for file in files[:-1]:
                        file.close()
                    temp=[]
                    for n in range(len(self.vars_)):
                        temp.append(open(filePath+"/"+str(n)+"."+str(int(iter_//(networksPerFile*samplingStep)))+".txt", "wb"))
                    files=temp+[files[-1]]
                #Record prediction
                if(iter_>startSampling and (iter_-1)%samplingStep==0):
                    if(returnPredictions):                    
                        self.results.append(result_)
                    if(filePath is not None):
                        for n in range(len(files)-1):    
                            np.savetxt(files[n],self.vars_[n])
        
        #Update the summary file            
        file=files[-1]
        for n in range(len(self.vars_)):
            val=""
            for sizes in self.vars_[n].shape:
                val+=str(sizes)+" "
            val=val.strip()+"\n"
            file.write(val.encode("utf-8"))
        numNetworks=(epochs-startSampling)//samplingStep
        numFiles=numNetworks//networksPerFile
        if(numNetworks%networksPerFile!=0):
            numFiles+=1
        file.write((str(numNetworks)+" "+str(numFiles)+" "+str(len(self.vars_))).encode("utf-8"))
        for file in files:
            file.close()
        if(returnPredictions):
            return(self.results)
