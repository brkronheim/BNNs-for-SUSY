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
    
    #@tf.function    
    def metrics(self, predictions, scaleExp, train, mean=1, sd=1):
        """Calculates the average squared error and percent difference of the 
        current network
        Arguments:
            * predictions: output from the network
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
        
        squaredError=tf.reduce_mean(input_tensor=tf.math.squared_difference(predictions, y))
        scaled=tf.add(tf.multiply(predictions, sd), mean)
        real=tf.add(tf.multiply(y, sd), mean)
        if(scaleExp):
            scaled=tf.exp(scaled)
            real=tf.exp(real)
        else:
            scaled=logits
        percentError=tf.reduce_mean(input_tensor=tf.multiply(tf.abs(tf.divide(tf.subtract(scaled, real), real)), 100))
        return(predictions, squaredError, percentError)
        
    @tf.function    
    def calculateProbs(self, *argv):
        """Calculates the log probability of the current network values
        as well as the log probability of their prediction.
        
        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases.
        Returns:
            * prob: log probability of network values and network prediction
        """

        
        prob=tf.reduce_sum(input_tensor=self.make_response_likelihood(argv).log_prob(self.trainY))
        #probability of the network parameters
        index=0
        #print(len(argv))
        for n in range(len(self.layers)):
            numTensors=self.layers[n].numTensors
            if(numTensors>0):    
                prob+=self.layers[n].calculateProbs(argv[index:index+numTensors])
                index+=numTensors
        #print("prob", prob)
        return(prob)
        
    @tf.function    
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
                                 self.states[index:index+numTensors])
                indexh+=numHyperTensors
                index+=numTensors
        return(prob)

    #@tf.function
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
            tensors=self.states
        else:
            tensors=tensors[0]
        x=self.trainX
        if(not train):
            x=self.validateX

        prediction=tf.transpose(a=x)
        index=0
        for n in range(len(self.layers)):
            numTensors=self.layers[n].numTensors
            prediction=self.layers[n].predict(prediction,tensors[index:index+numTensors])
            index+=numTensors
        return(prediction)
    

    def add(self, layer, parameters=None):
        """Adds a new layer to the network
        Arguments:
            * layer: the layer to be added
            * weigths: matrix to initialize weights
            * biases: matrix to initialize biases
        """
        self.layers.append(layer)
        if(layer.numTensors>0):
            if parameters is None:
                for states in layer.parameters:
                    self.states.append(states)
            else:
                for states in parameters:
                    self.states.append(states)

        if(layer.numHyperTensors>0):        
            for states in layer.hypers:
                self.hyperStates.append(states)
            
            
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
        self.cores=cores
        
        self.hyper_step_size = tf.Variable(np.array(hyperStepSize, self.dtype))
        
        #Setup the Markov Chain for the network parameters
        self.mainKernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
                target_log_prob_fn=self.calculateProbs, #used to calculate log density
                num_leapfrog_steps=self.leapfrog,
                step_size=self.step_size,
                step_size_update_fn=None, 
                state_gradients_are_stopped=True)

        
        #Setup the Transition Kernel for the hyper parameters
        self.hyperKernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
                target_log_prob_fn=self.calculateHyperProbs, #used to calculate log density
                num_leapfrog_steps=hyperLeapfrog,
                step_size=self.hyper_step_size,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=50, decrement_multiplier=0.01), 
                state_gradients_are_stopped=True)
        
    def updateStates(self):
        indexh=0
        index=0
        for n in range(len(self.layers)):
            numHyperTensors=self.layers[n].numHyperTensors
            numTensors=self.layers[n].numTensors
            if(numHyperTensors>0):    
                self.layers[n].updateHypers(self.hyperStates[indexh:indexh+numHyperTensors])
                indexh+=numHyperTensors
            if(numTensors>0):
                self.layers[n].updateParameters(self.states[index:index+numHyperTensors])
                index+=numTensors
               
    def updateKernels(self):
        self.step_size, self.leapfrog = self.adapt.update(self.states)
        #Setup the Markov Chain for the network parameters
        self.mainKernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
                target_log_prob_fn=self.calculateProbs, #used to calculate log density
                num_leapfrog_steps=self.leapfrog,
                step_size=self.step_size,
                step_size_update_fn=None, 
                state_gradients_are_stopped=True)

    #@tf.function
    def stepMCMC(self):
        num_results=1
        #Setup the Markov Chain for the network parameters
        self.states, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0, #start collecting data on first step
            current_state=self.states, #starting parts of chain 
            parallel_iterations=self.cores, 
            kernel=self.mainKernel)
        self.avg_acceptance_ratio = tf.reduce_mean(
            input_tensor=tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)))
        self.loss = -tf.reduce_mean(input_tensor=kernel_results.accepted_results.target_log_prob)
        for x in range(len(self.states)):
            self.states[x]=self.states[x][0]

        index=0
        for n in range(len(self.layers)):
            numTensors=self.layers[n].numTensors
            if(numTensors>0):
                self.layers[n].updateParameters(self.states[index:index+numTensors])
                index+=numTensors
            
            
        #Setup the Markov Chain for the hyper parameters
        self.hyperStates, hyper_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0, #start collecting data on first step
            current_state=self.hyperStates, #starting parts of chain 
            parallel_iterations=self.cores,
            kernel=self.hyperKernel)
        self.hyper_avg_acceptance_ratio = tf.reduce_mean(
            input_tensor=tf.exp(tf.minimum(hyper_kernel_results.log_accept_ratio, 0.)))
        self.hyper_loss = -tf.reduce_mean(input_tensor=hyper_kernel_results.accepted_results.target_log_prob)
        for x in range(len(self.hyperStates)):
            self.hyperStates[x]=self.hyperStates[x][0]
        
        indexh=0
        for n in range(len(self.layers)):
            numHyperTensors=self.layers[n].numHyperTensors
            if(numHyperTensors>0):    
                self.layers[n].updateHypers(self.hyperStates[indexh:indexh+numHyperTensors])
                indexh+=numHyperTensors
        
        
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
            for n in range(len(self.states)):
                files.append(open(filePath+"/"+str(n)+".0"+".txt", "wb"))
            files.append(open(filePath+"/summary.txt", "wb"))
        
        if(returnPredictions):
            self.results=[]
        logits=self.predict(False, self.states) #prediction placeholder
        #get a prediction, squared error, and percent error
        
        iter_=0
        tf.random.set_seed(50)
        while(iter_<epochs): #Main training loop
            #check that the vars are not tensors
            self.stepMCMC()
            
            result, squaredError, percentError=self.metrics(self.predict(train=False), scaleExp, False, mean, sd)
            
            
            iter_+=1
            
            print()
            
            print('iter:{:>2}  Network loss:{: 9.3f}  step_size:{:.7f} leapfrog_num:{:>4} avg_acceptance_ratio:{:.4f}'.format(
                      iter_, self.loss, self.step_size, self.leapfrog, self.avg_acceptance_ratio))
            print('Hyper loss:{: 9.3f}  step_size:{:.7f}  avg_acceptance_ratio:{:.4f}'.format(
                            self.hyper_loss*1, self.hyper_step_size*1, self.hyper_avg_acceptance_ratio*1))
            print('squaredError{: 9.5f} percentDifference{: 7.3f}'.format(squaredError, percentError))
            
            self.updateKernels()
            
            #self.step_size, self.leapfrog=self.adapt.update(nextStates)
            
            #Create new files to record network
            if(iter_>startSampling and (iter_-1-startSampling)%(networksPerFile*samplingStep)==0):
                for file in files[:-1]:
                    file.close()
                temp=[]
                for n in range(len(self.states)):
                    temp.append(open(filePath+"/"+str(n)+"."+str(int(iter_//(networksPerFile*samplingStep)))+".txt", "wb"))
                files=temp+[files[-1]]
            #Record prediction
            if(iter_>startSampling and (iter_-1)%samplingStep==0):
                if(returnPredictions):                    
                    self.results.append(result_)
                if(filePath is not None):
                    for n in range(len(files)-1):    
                        np.savetxt(files[n],self.states[n])

        #Update the summary file            
        file=files[-1]
        for n in range(len(self.states)):
            val=""
            for sizes in self.states[n].shape:
                val+=str(sizes)+" "
            val=val.strip()+"\n"
            file.write(val.encode("utf-8"))
        numNetworks=(epochs-startSampling)//samplingStep
        numFiles=numNetworks//networksPerFile
        if(numNetworks%networksPerFile!=0):
            numFiles+=1
        file.write((str(numNetworks)+" "+str(numFiles)+" "+str(len(self.states))).encode("utf-8"))
        for file in files:
            file.close()
        if(returnPredictions):
            return(self.results)
