import click

import numpy as np
import tensorflow as tf

from scipy import stats
from BNN_functions import normalizeData, build_input_pipeline, runInference

@click.command()
@click.option('--name', help='Name of network to load')
@click.option('--out', default="test", help='Name of file to which analysis is written')
@click.option('--iters', default=100, help='Number of iterations over the input data')

def main(name, out, iters):
    """This script loads a previously created neural network created in
    BNN_train. It will then run the validation data through the network the
    number of times specified in --iters. This data will be collected and fit
    to a normal distribution which can be compared to the actual data point.
    From this analysis the script will print the percent of the real data that
    falls within 1, 2, and 3 standard deviations of the predicted distributions.
    Finally, in the file is given in --out  or test.txt the script will write 
    the unnormalized mean, lower 95% confidence value, higher 95% confidence 
    value, the width of the 95% confidence interval, and the actual value. It
    will also print out the percent below the minimum value of the distribution
    and the percent above the maximum value of the distribution.
    """

    trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
    trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
    valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
    valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)


    #Normalize the training and output data and collect the values used to do so
    normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut)
        
    with tf.Session() as sess:
        
        #Load the previously created network
        print('\nLoading...')
        logits = tf.get_variable("logits",[])
        saver = tf.train.import_meta_graph(name+'.meta')
        saver.restore(sess, './'+name)
        graph = tf.get_default_graph()
        logits = graph.get_tensor_by_name("bayesian_neural_net/logits:0")
        rate = graph.get_tensor_by_name("rate:0")
        handle = graph.get_tensor_by_name("handle:0")
        print('Ok')
        
        #values required to run network
        batch_size=128 
        dropoutPercent=0.0
        
        #size of data
        train_size=len(trainIn[:,1])
        val_size=len(valIn[:,1])
        data_size=train_size+val_size
        
        #setup data pipelines
        (x_input, y_output, handle, training_iterator, validation_iterator) = build_input_pipeline(
           data, train_batch, randomize = False, handle=handle)
        
        #prediction operator
        pred = runInference(logits)
        
        #iterator handles
        train_handle = sess.run(training_iterator.string_handle())
        validate_handle = sess.run(validation_iterator.string_handle())
        
        #iterate over the input data several times
        allPredictions=np.zeros((iters,len(data[3])))
        steps=iters*len(data[3])
        decile=steps
        if(steps/10>=1):
            decile=int(steps/10)
        total=0
        while(total<steps):
            prediction= sess.run([pred], feed_dict={rate: dropoutPercent, handle: validate_handle})
            prediction=prediction[0]
            
            for j in range(len(prediction)):
                if(total<iters*len(data[3])):
                    allPredictions[int(total/len(data[3]))%iters,(total)%(len(data[3]))]=prediction[j]
                total+=1    
                if(total%decile==0):
                    print("{:.2f} percent of data generated".format(100*total/steps))
        
        real=data[3]*normInfo[0][1]+normInfo[0][0]
        wrong=[]
        sd3=[]
        sd2=[]
        sd1=[]
        belMin=[]
        abvMax=[]
        percentError=[]
        name=out+".txt"
        allPredictions=allPredictions*normInfo[0][1]+normInfo[0][0]
        with open(name, 'w') as file:
            decile=int(len(allPredictions[0,:])/10)
            for k in range(len(allPredictions[0,:])):
                #fit output distribution
                minimum=min(allPredictions[:,k])
                maximum=max(allPredictions[:,k])
                mean, sd = stats.norm.fit(allPredictions[:,k])
                
                #calculate the unnormalized values at each of the standard deviations
                low99=mean-sd*3
                low95=mean-sd*2
                low68=mean-sd
                high68=mean+sd
                high95=mean+sd*2
                high99=mean+sd*3
                actual=real[k]
                
                expLow=np.exp(low95)
                expHigh=np.exp(high95)
                expMean=np.exp(mean)
                expActual=np.exp(actual) 
                #write data to the output file
                file.write(str(expMean) + "\t" + str(expLow) + "\t" + 
                           str(expHigh) + "\t" + str(expHigh - expLow) + "\t" + 
                           str(expActual) + "\n")
                
                #Compare values to distribution max and min
                if(actual<minimum):
                    belMin.append(k)
                elif(actual>maximum):
                    abvMax.append(k)
                
                
                #Find out where the actual data point falls in the output distribtuion
                if(actual<low99 or actual>high99):
                    wrong.append(k)
                elif(actual<low95):
                    sd3.append(k)
                elif(actual<low68):
                    sd2.append(k)
                elif(actual<high68):
                    sd1.append(k)
                elif(actual<high95):
                    sd2.append(k)
                elif(actual<=high99):
                    sd3.append(k)
                if((k+1)%decile==0):
                    print("{:.2f} percent of data analyzed".format(100*(k+1)/len(allPredictions[0,:])))
                    
            print("Number outside of 3 standard deviations:", len(wrong))
            print("Number between 2 and 3 standard deviations:", len(sd3))
            print("Number between 1 and 2 standard deviations:", len(sd2))
            print("Number inside 1 standard deviation:", len(sd1))
            print("Number below distribution minimum:", len(belMin))
            print("Number above distribution maximum:", len(abvMax))
            print()
            print("Percent inside 1 standard deviation:", 100*len(sd1)/len(allPredictions[0,:]))
            print("Percent inside 2 standard deviations:",100*(len(sd1)+len(sd2))/len(allPredictions[0,:]))
            print("Percent inside 3 standard deviations:",100*(len(sd1)+len(sd2)+len(sd3))/len(allPredictions[0,:]))
            print("Percent outside 3 standard deviations:", 100*len(wrong)/len(allPredictions[0,:]))
            print("Percent below distribution minimum:", 100*len(belMin)/len(allPredictions[0,:]))
            print("Percent above distribution maximum:", 100*len(abvMax)/len(allPredictions[0,:]))

if(__name__ == "__main__"):
    main()
