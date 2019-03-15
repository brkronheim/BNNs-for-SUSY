# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:29:02 2019

@author: brade
"""
import pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from BNN_functions import normalizeData

#import layer
import network

tfd = tfp.distributions



trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)


#Normalize the training and output data and collect the values used to do so
normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 



inputDims=1
hiddenDims=20
hiddenLayers=2
outputDims=1
dtype=np.float32


num_samples_train=50
num_samples_val=100
numNetworks=1
num_iters = int(10)

x=np.linspace(-1,1,num_samples_train)
y=np.array([np.sin(x*2)/2+np.cos(x*5)/2])
testx=np.linspace(-1,1,num_samples_val)
testy=np.array([np.sin(testx*2)/2+np.cos(testx*5)/2])


results=[]

neuralNet=network.network(20, inputDims,hiddenDims,hiddenLayers,outputDims,dtype,x,y)


with tf.Session() as sess:
    step_size = tf.Variable(np.array(0.0001, dtype))
    hyper_step_size = tf.Variable(np.array(0.001, dtype))
    num_results = 2 #number of markov chain draws
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0, #start collecting data on first step
        current_state=neuralNet.states, #starting parts of chain 
        kernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
            target_log_prob_fn=neuralNet.calculateProbs, #used to calculate log density
            num_leapfrog_steps=20,
            step_size=step_size,
            
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(decrement_multiplier=0.01), 
            state_gradients_are_stopped=True))
    avg_acceptance_ratio = tf.reduce_mean(
        tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)))
    loss = -tf.reduce_mean(kernel_results.accepted_results.target_log_prob)
    
    
    hyper_states, hyper_kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0, #start collecting data on first step
        current_state=neuralNet.hyperStates, #starting parts of chain 
        kernel=tfp.mcmc.HamiltonianMonteCarlo( #use HamiltonianMonteCarlo to step in the chain
            target_log_prob_fn=neuralNet.calculateHyperProbs, #used to calculate log density
            num_leapfrog_steps=20,
            step_size=hyper_step_size,
            
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(decrement_multiplier=0.01), 
            state_gradients_are_stopped=True))
    hyper_avg_acceptance_ratio = tf.reduce_mean(
        tf.exp(tf.minimum(hyper_kernel_results.log_accept_ratio, 0.)))
    hyper_loss = -tf.reduce_mean(hyper_kernel_results.accepted_results.target_log_prob)
    
    
    
    
    init_op = tf.global_variables_initializer()
    init_op.run()

    
    vars_=neuralNet.vars_
    hyper_vars=neuralNet.hyperVars

    iter_=0
    
    while(iter_<1000):
        for n in range(len(vars_[iter_%20])):
            if(tf.contrib.framework.is_tensor(vars_[iter_%20][n])):
                vars_[iter_%20][n]=sess.run(vars_[iter_%20][n])
        [
            nextStates,
            loss_,
            step_size_,
            avg_acceptance_ratio_,
        ] = sess.run([
            states,
            loss,
            step_size,
            avg_acceptance_ratio,
        ], feed_dict={tuple(neuralNet.states): tuple(vars_[iter_%20])})
        for n in range(len(vars_[iter_%20])):
            vars_[(1+iter_)%20][n]=nextStates[n][-1]
        iter_+=1
        print('iter:{:>2}  Network loss:{: 9.3f}  step_size:{:.7f}  avg_acceptance_ratio:{:.4f}'.format(
                  iter_, loss_, step_size_, avg_acceptance_ratio_))
        neuralNet.currentNetworksVars=vars_[(iter_)%20][n]
        hyper_vars=neuralNet.hyperVars
        if(iter_%50==0):
            hyperIter=0
            while(hyperIter<20):
                for n in range(len(hyper_vars[iter_%20])):
                    if(tf.contrib.framework.is_tensor(hyper_vars[hyperIter%20][n])):
                        hyper_vars[hyperIter%20][n]=sess.run(hyper_vars[hyperIter%20][n])
                [
                    nextHyperStates,
                    hyperLoss_,
                    hyper_step_size_,
                    hyper_avg_acceptance_ratio_,
                ] = sess.run([
                    hyper_states,
                    hyper_loss,
                    hyper_step_size,
                    hyper_avg_acceptance_ratio,
                ], feed_dict={tuple(neuralNet.hyperStates): tuple(hyper_vars[hyperIter%20])})
                for n in range(len(hyper_vars[hyperIter%20])):
                    hyper_vars[(1+hyperIter)%20][n]=nextHyperStates[n][-1]
                hyperIter+=1
                print('iter:{:>2}  Hyper loss:{: 9.3f}  step_size:{:.7f}  avg_acceptance_ratio:{:.4f}'.format(
                        hyperIter, hyperLoss_, hyper_step_size_, hyper_avg_acceptance_ratio_))
                if(hyperIter==20):
                    neuralNet.update(hyper_vars[(hyperIter)%20])
        
        if(iter_>500 and iter_%50==0):
            squaredError, percentError=sess.run(neuralNet.metrics(2,1, vars_[(iter_-1)%20]))
            print('squaredError{: 9.5f} percentDifference{: 7.3f}'.format(squaredError, percentError))
            neuralNet.x = tf.reshape(tf.constant(testx, dtype=neuralNet.dtype),[len(testx),1])
            neuralNet.y = tf.constant(testy, dtype=neuralNet.dtype)
            out=neuralNet.predict(vars_[(iter_-1)%20])
            results.append(sess.run(out))
            neuralNet.x = tf.reshape(tf.constant(x, dtype=neuralNet.dtype),[len(x),1])
            neuralNet.y = tf.constant(y, dtype=neuralNet.dtype)

numNetworks=len(results)

plt.figure(1)
for n in range(numNetworks):
    plt.plot(testx, results[n][0])

average=[None]*len(testy[0])
for n in range(len(testy[0])):
    val=0
    for m in range(numNetworks):
        val+=results[m][0][n]/numNetworks
    average[n]=val
plt.plot(testx, testy[0], label="real")
plt.plot(testx, average, label="average")
plt.ylim(-2,2)
plt.legend()

a=np.array(results)
mean=[]
low95=[]
high95=[]
percentDifference=0
for n in range(num_samples_val):
    m=np.mean(a[:,0,n])
    s=np.std(a[:,0,n])
    mean.append(m)
    low95.append(m-2*s)
    high95.append(m+2*s)
    percentDifference+=min(np.abs((m-testy[0][n])*100/(testy[0][n]))/(num_samples_val),100/(num_samples_val))    

plt.figure(2)
plt.plot(testx, mean, label="mean")
plt.plot(testx,low95, label="lower 95% confidence")
plt.plot(testx,high95, label="upper 95% confidence")
plt.plot(testx, testy[0], label="real")
plt.legend()

plt.figure(3)
plt.plot(testx, mean, label="mean")
plt.plot(testx,low95, label="lower 95% confidence")
plt.plot(testx,high95, label="upper 95% confidence")
plt.plot(testx, testy[0], label="real")
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend()
print(percentDifference)
