# BNNs-for-SUSY
This repository contains code which can be used to create Bayesian Neural Networks in order to make predictions of the cross sections of supersymmetric particles with them. The folder Weekly-Reports contains the pdfs for presentations during weekly research meetings. The Python and Shell scripts used for generating and parsing training data are included in the folder generateData. Code to train and use a probabilistic neural network built on the layers in Tensorflow-Probability is in the folder neuralNetworks, and the code to train true BNNs is located in the folder bayesianNetwork.

## Dependencies
All python code written here is intended to be used in Python3. 
### generateData
The code in generateData is depended upon the Python package Numpy and the Shell tool GNU parallel. Both of these can be installed through the Linux command line. There are further software requirements, but these are slightly more involved and described in the readme within the generateData folder.

Numpy can be installed through the command:
```
pip3 install numpy
```
GNU parallel can be installed through the command:
```
sudo apt-get install parallel
```

### neuralNetworks
The code here is dependent upon numpy, tensorflow, tensorflow-probability, scipy, click, and pandas. These can be installed through the following command:
```
pip3 install numpy tensorflow-probability scipy click pandas
```
Tensorflow can be installed for just cpu usage with:
```
pip3 install tensorflow
```
It can also be installed for gpu usage with:
```
pip3 install tensorflow-gpu
```

### bayesianNetwork
This code is dependent upon the same pacakges as nerualNetworks.

## Usage
The full usage details for each of the folders is contained within their respective readmes. Included here is a brief explanation of what the code in each folder can do.

### generateData
The shell script makeData.sh within generateData will run the programs Prospino and SUSY-HIT to generate the cross section for creating a supersymmetric particle within the LHC according to the phenomenological Minimal Supersymmetric Standard Model. The python script createDataSets.py can then be used to create training sets from this data for machine learning purposes. 

### neuralNetworks
The code within this folder can be used to train the probabilistic neural networks using tensorflow-probability. Specifically, this code trains a neural network with normal distributions for each of the weights and biases. This means that the network will almost always produce a slightly different result every time it is called, even for the same input values. That means that through making predictions many times on the same data, a distribution of results can be obtained. Optimally, the spread of this distribution should reflect the certainty of the network about the quality of the prediction. Unfortunately, the network tends to have tighter distributions that it really ought to.

### bayesianNetwork
Inside this folder is code that will train a true Bayesain neural network. This machine learning algorithm uses the Hamlitonain Monte Carlo algorithm to find the posterior distribution of neural networks given the training data and some assumptions about the distributions of weights and biases. The result is an ensemble of neural networks which should reflect the broad array of different neural networks whcih can be good. These networks will also give a distribution of results for a single point and outperform in almost every way the networks train in the neuralNetworks folder. The only downside to this training method is that it is much more computionally intensive. 
