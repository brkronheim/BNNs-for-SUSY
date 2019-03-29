import warnings
import os

import pylab as plt
import numpy as np

from scipy import stats
from BNN_functions import normalizeData

from layer import DenseLayer
from activationFunctions import Relu
import network

warnings.filterwarnings("ignore", category=DeprecationWarning) 


trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)


#Normalize the training and output data and collect the values used to do so
normInfo, data = normalizeData(trainIn, trainOut, valIn, valOut) 

inputDims=19
hiddenDims=50
hiddenLayers=2
dtype=np.float32


neuralNet=network.network(dtype,data[0],data[1], data[2], data[3])
neuralNet.add(DenseLayer(inputDims,hiddenDims))
for n in range(hiddenLayers-1):
    neuralNet.add(DenseLayer(hiddenDims, hiddenDims))
    neuralNet.add(Relu())
neuralNet.add(DenseLayer(hiddenDims, outputDims))
neuralNet.setupMCMC(0.000001, 0.01, 20, 20)
savedResults=neuralNet.train(10000,50,10,500,100, normInfo[0][0], normInfo[0][1])


results=np.array(savedResults)
print(results.shape)    
real=data[3]*normInfo[0][1]+normInfo[0][0]
wrong=[]
sd3=[]
sd2=[]
sd1=[]
belMin=[]
abvMax=[]
percentError=[]
results=results*normInfo[0][1]+normInfo[0][0]
decile=int(len(results[0,0,:])/10)
for k in range(len(results[0,0,:])):
    
    #fit output distribution
    minimum=min(results[:,0,k])
    maximum=max(results[:,0,k])
    mean, sd = stats.norm.fit(results[:,0,k])
    
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

    percentError.append(100*abs(expMean-expActual)/(expActual))
    
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
        print("{:.2f} percent of data analyzed".format(100*(k+1)/len(results[0,:])))
        plt.figure(k)
        plt.hist(results[:,0,k],color="b", bins=50)

        plt.axvline(x=low95,color="r")
        plt.axvline(x=low68,color="y")
        plt.axvline(x=mean,color="k")
        plt.axvline(x=high68,color="y")
        plt.axvline(x=high95,color="r")
        plt.axvline(x=actual,color="g")
        plt.legend(["-2 SD", "-1 SD", "Mean", "1 SD", "2 SD", "Actual"])
        plt.show();

mean, sd = stats.norm.fit(percentError)
plt.figure(20)
plt.hist(percentError,color="b", bins=150, range=(0,150))
plt.title("Percent Error")
plt.show()

print("Percent Error Mean:", mean)
print("Percent Error Standard Deviation:", sd)
print()


print("Number outside of 3 standard deviations:", len(wrong))
print("Number between 2 and 3 standard deviations:", len(sd3))
print("Number between 1 and 2 standard deviations:", len(sd2))
print("Number inside 1 standard deviation:", len(sd1))
print("Number below distribution minimum:", len(belMin))
print("Number above distribution maximum:", len(abvMax))
print()
print("Percent inside 1 standard deviation:", 100*len(sd1)/len(results[0,:]))
print("Percent inside 2 standard deviations:",100*(len(sd1)+len(sd2))/len(results[0,:]))
print("Percent inside 3 standard deviations:",100*(len(sd1)+len(sd2)+len(sd3))/len(results[0,:]))
print("Percent outside 3 standard deviations:", 100*len(wrong)/len(results[0,:]))
print("Percent below distribution minimum:", 100*len(belMin)/len(results[0,:]))
print("Percent above distribution maximum:", 100*len(abvMax)/len(results[0,:]))
