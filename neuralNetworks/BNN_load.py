import click

import numpy as np
import tensorflow as tf

from scipy import stats
from BNN_Functions import normalizeData, build_input_pipeline, runInference

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
    value, the width of the 95% confidence interval, and the actual value.    
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
        (x_input, y_output, handle, training_iterator, validation_iterator, full_iterator) = build_input_pipeline(
           data, batch_size, val_size, data_size, handle)
        actual, pred = runInference(y_output, logits)
        allPredictions=[]
        full_handle = sess.run(full_iterator.string_handle())

        #iterate over the input data several times
        for i in range(iters):
            real, prediction = sess.run([actual, pred], feed_dict={rate: dropoutPercent, handle: full_handle})
            allPredictions.append(prediction)
            
        
        #data storage for predictions
        allPredictions=np.array(allPredictions)
        real=data[3]
        wrong=[]
        sd3=[]
        sd2=[]
        sd1=[]
        #start writing to the outpuf file
        name=out+".txt"
        with open(name, 'w') as file:
            for k in range(len(allPredictions[0,:])):
                #fit output distribution
                mean, sd = stats.norm.fit(allPredictions[:,k]*normInfo[0][1]+normInfo[0][0])
                
                #calculate the unnormalized values at each of the standard deviations
                low99=np.exp(mean-sd*3)
                low95=np.exp(mean-sd*2)
                low68=np.exp(mean-sd)
                high68=np.exp(mean+sd)
                high95=np.exp(mean+sd*2)
                high99=np.exp(mean+sd*3)
                actual=np.exp(real[k]*normInfo[0][1]+normInfo[0][0])
                
                #write data to the output file
                file.write(str(np.exp(mean)) + "\t" + str(low95) + "\t" + 
                           str(high95) + "\t" + str(high95 - low95) + "\t" + 
                           str(actual) + "\n")
                
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
            print("Number outside of 3 standard deviations:", len(wrong))
            print("Number between 2 and 3 standard deviations:", len(sd3))
            print("Number between 1 and 2 standard deviations:", len(sd2))
            print("Number inside 1 standard deviation:", len(sd1))
            print("Percent inside 1 standard deviation:", 100*len(sd1)/len(allPredictions[0,:]))
            print("Percent inside 2 standard deviations:",100*(len(sd1)+len(sd2))/len(allPredictions[0,:]))
            print("Percent inside 3 standard deviations:",100*(len(sd1)+len(sd2)+len(sd3))/len(allPredictions[0,:]))
            print("Percent outside 3 standard deviations:", 100*len(wrong)/len(allPredictions[0,:]))

if(__name__ == "__main__"):
    main()