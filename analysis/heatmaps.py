import tensorflow as tf
import pylab as plt
import numpy as np
import seaborn as sns

import matplotlib.ticker as ticker

import os

def predict(inputVal, numNetworks, numMatrices, matrices, skip=10):
    """Generates a prediction from inputVal
        
    Arguments:
        * inputVal: input parameters, can be more than one
        * numNetworks: total number of networks saved
        * numMatrices: number of matrices in the network
        * matrices: matrices used to make the prediction
        * skip: networks are choosen in increments of skip
    Returns:
        * mean: vector of average output across all networks for each input
        * sd: vector of standard deviations across all networks for each input
    
    """
    
    inputVal=tf.transpose(inputVal)
    results=[None]*(numNetworks//skip)
    for m in range(0,numNetworks,skip):
        current=inputVal
        for n in range(0,numMatrices,2):
            current=tf.matmul(matrices[n][m,:,:],current)
            current+=matrices[n+1][m,:,:]
            if(n+2<numMatrices):
                current=tf.maximum(current,0)
        results[m//skip]=current[0]
    mean=1*tf.reduce_mean(results,axis=0)
    sd=1*tf.math.reduce_std(results, axis=0)
    return(mean,sd)

def genStart(index1, index2, number, samples, dtype=np.float32):
    """Generates starting points for each heatmap. Parameters 1 and 6 are held
    at 1.0 unless they are the one being varied. Other parameter values are random but
    constant throughout the heatmap.
    
    Arguments:
        * index1: first parameter to be varied
        * index2: second parameter to be varied
        * number: number of values index1 and index2 should have
        * samples: number of random samples for each combination of index1 and 2
        * dtype: data type

    Returns:
        * points1: Each value the varied parameters take on
        * params: A matrix with all of the parameter values
        
    """
    points1=np.linspace(-1,1,number)
    points2=np.linspace(-1,1,number)
    vals=np.random.uniform(-1,1,(19,samples))
    filler=tf.ones((samples))
    collection=[]
    for point1 in points1:
        for point2 in points2:
            temp=vals
            temp[1,:] = 1.0*filler
            temp[6,:] = 1.0*filler
            temp[index1,:] = point1*filler
            temp[index2,:] = point2*filler
            collection.append(np.float32(temp))
    params=tf.concat(collection,1)
    print(params.shape)
    return(points1, params)

def main():

    #Load the networks
    directory="/home/kronheim/BNNs/BNNs-for-SUSY/bayesianNetwork2.0/StackedNadamStart5Deep50Wide/"
    summary=[(50,19),(50,1),(50,50),(50,1),(50,50),(50,1),(50,50),(50,1),(50,50),(50,1),(1,50),(1,1),(0,0)]
    numMatrices=12
    numNetworks=50
    numFiles=12
    matrices=[]
    for n in range(numMatrices):
        weightsSplitDims=(numNetworks*numFiles,int(summary[n][0]),int(summary[n][1]))
        weights0=np.zeros(weightsSplitDims)
        for m in range(numFiles):
            weights=np.loadtxt(directory+str(n)+"."+str(m)+".txt", dtype=np.float32,ndmin=2)
            for k in range(numNetworks):
                weights0[m*numNetworks+k,:,:]=weights[weightsSplitDims[1]*k:weightsSplitDims[1]*(k+1),:weightsSplitDims[2]]
        matrices.append(tf.cast(weights0, tf.float32))
    numNetworks*=numFiles
    
    
    samplePoints=100 #Number of points across each varied dimension
    sampleDepth=1000 #Number of random states to calcualte for each point in the heatmap
    
    #Generate the heatmaps
    for n in range(19):
        for m in range(19):
            points, params = genStart(n, m, samplePoints, sampleDepth) #Get starting points
            crossSections, sd=predict(tf.transpose(tf.constant(params,tf.float32))) #Get cross sections
            crossSections=tf.reshape(crossSections,(samplePoints,samplePoints, sampleDepth))
            print(crossSections.shape)
            means=tf.reduce_mean(crossSections,axis=-1)
            sds=tf.math.reduce_std(crossSections,axis=-1)

            directory=os.getcwd()+"/Heatmaps"

            def func(a,b):
                """Used for tick locations"""
                return(str((a-50)/50))

            #Create heatmap
            plt.figure()        
            ax=sns.heatmap(means, linewidth=0.00, xticklabels=points, yticklabels=points)

            ax.xaxis.set_major_locator(ticker.LinearLocator(11))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))

            ax.yaxis.set_major_locator(ticker.LinearLocator(11))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(func))


            plt.title("Parameters " + str(n) + " by " + str(m) + " mean")
            plt.xlabel("Param "+str(m))
            plt.ylabel("Param "+str(n))
            plt.tight_layout()
            plt.savefig(directory+"/Parameters_" + str(n) + "_by_" + str(m) + "_mean.png")
            plt.close()

            plt.figure()
            ax=sns.heatmap(sds, linewidth=0.00)


            ax.xaxis.set_major_locator(ticker.LinearLocator(11))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))

            ax.yaxis.set_major_locator(ticker.LinearLocator(11))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(func))


            plt.title("Parameters " + str(n) + " by " + str(m) + " sd")
            plt.xlabel("Param "+str(m))
            plt.ylabel("Param "+str(n))
            figure = ax.get_figure()    
            plt.tight_layout()
            plt.savefig(directory+"/Parameters_"  + str(n) + "_by_" + str(m) + "_sd.png")
            plt.close()

        
if(__name__=="__main__"):
    main()
        
        
