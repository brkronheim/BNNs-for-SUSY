import pylab as plt
import numpy as np
from BNN_functions import normalizeData

from layer import DenseLayer
import network

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


num_samples_train=10
num_samples_val=100
numNetworks=1
num_iters = int(10)

x=np.linspace(-1,1,num_samples_train)
y=np.array([np.sin(x*2)/2+np.cos(x*5)/2])
testx=np.linspace(-1,1,num_samples_val)
testy=np.array([np.sin(testx*2)/2+np.cos(testx*5)/2])



neuralNet=network.network(inputDims,hiddenDims,hiddenLayers,outputDims,dtype,x,y,testx, testy)
neuralNet.add(DenseLayer(inputDims,hiddenDims))
for n in range(hiddenLayers-1):
    neuralNet.add(DenseLayer(hiddenDims, hiddenDims))
neuralNet.add(DenseLayer(hiddenDims, outputDims)) 
neuralNet.setup()
results=neuralNet.train(100,10,10,50,10)
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
    percentDifference+=np.abs((m-testy[0][n])*100/(testy[0][n]+2))/(num_samples_val)   

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
