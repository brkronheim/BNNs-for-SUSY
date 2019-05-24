## Usage
Through the use of the objects contained within this folder it is possible to easily make Bayesian Neural Networks for regression learning problems. The file train.py contains an excellent example with click implemented for ease of use in the command line. More generally, in order to use this code you must import network, Dense Layer, and an activation such as Relu. This can be done as follows:
```
from layer import DenseLayer
import network
from activationFunctions import Relu
```
Next, it is highly convenient to turn off the deprecation warnings. These are all from tensorflow, tensorflow-probability, and numpy intereacting with tensorflow, so it isn't something easily fixed and there are a lot of warnings. These are turned off with:
```
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
```
The other important setup task is determining whether or not to use the GPU. Due to the ability for many parts of this code to run in parallel, it is better to run this code on 10+ CPU cores than a single GPU core. If this situation is applicable, the following code tells tensorflow to only use CPUs:
```
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```
Moving on to the actual use of this code, start with the declaration of a network object:
```
neuralNet = network.network(dtype, inputDims, trainX, trainY, validationX, validationY, mean, sd)
```
The paramaters are described as follows:
* dtype: data type for Tensors
* inputDims: dimension of input vector
* trainX: the training data input, shape is n by inputDims
* trainY: the training data output
* validateX: the validation data input, shape is n by inputDims
* validateY: the validation data output
* mean: the mean used to scale trainY and validateY
* sd: standard deviation used to scale trainY and validateY

Next, add all of the desired layers and activation functions as follows:
```
neuralNet.add(DenseLayer(inputDims,width, seed=seed))
neuralNet.add(Relu())
```
The paramater inputDims is the output shape of the layer before, and the width is the ouput shape of the layers itself. The seed is used for seeding the random number generator. There are other activation functions, but currently on Relu can be used when saving, so it is the recomended choice.

Next, the Markov Chain Monte Carlo algorithm must be initialized. This can be done as follows:
```
neuralNet. setupMCMC(self, stepSize, stepMin, stepMax, leapfrog, leapMin, leapMax,
                  hyperStepSize, hyperLeapfrog, burnin, cores):
```
This is a description of the paramaters:m 
* stepSize: the starting step size for the weights and biases
* stepMin: the minimum step size
* stepMax: the maximum step size
* leapfrog: number of leapfrog steps for weights and biases
* leapMin: the minimum number of leapfrog steps
* leapMax: the maximum number of leapfrog steps
* hyperStepSize: the starting step size for the hyper parameters
* hyperLeapfrog: leapfrog steps for hyper parameters
* cores: number of cores to use

This code uses the adaptive Hamlitonain Monte Carlo described in "Adaptive Hamiltonian and Riemann Manifold Monte Carlo Samplers" by Wang, Mohamed, and de Freitas. In accordance with this paper there are a few more paramaters that can be adjusted by making changes to the network.py file and the paramAdapter.py file. The paramAdapter object is initialized in line 217 of network.py with the call 
```
paramAdapter(e1,L1,el,eu,Ll,Lu,m,k,a=4,delta=0.1,cores=4)
```
These fields are described as:
* e1: starting step size
* L1: starting number of leapfrog steps
* el: lower step size bound
* eu: upper step size bound
* Ll: lower leapfrog bound
* Lu: upper leapfrog bound
* m: number of averaging steps
* k: iterations before proposal probability starts decreasing
* a: constant, 4 in paper
* delta: constant, 0.1 in paper
* cores: number of cores to use in processing
Currently, m is set as 2, and k is set as the number of burnin steps divided by k. Both of these values can be changed.
Additionally, the paramAdapter object creates a grid to search possible step sizes and numbers of leapfrog steps. The program currently checks 500 distinct step size values. This can be changed to a different value by changing the 500 in line 37 of paramAdapter.py


The last thing to do is actually tell the model to start learning this is done with the following command:
```
def train(self, epochs, startSampling, samplingStep, mean=0, sd=1, scaleExp=False, folderName=None, 
          networksPerFile=1000, returnPredictions=False):
```
The arguments have the following meanings:

* Epochs: Number of training cycles
* startSampling: Number of epochs before networks start being saved
* samplingStep: Epochs between sampled networks
* mean: true mean of output distribution
* sd: true sd of output distribution
* scaleExp: whether the metrics should be scaled via exp
* folderName: name of folder for saved networks
* networksPerFile: number of networks saved in a given file
* returnPredictions: whether to return the prediction from the network

Once the network has trained, which may take a while, the saved networks can be loaded and then used to make predictions using the following code:
```
import os

from BNN_functions import normalizeData, loadNetworks, predict

numNetworks, numMatrices, matrices=loadNetworks(filePath)

initialResults = predict(inputData, numNetworks, numMatrices, matrices)
```
The variable filePath is the directory from which the networks are being loaded, and inputData is the data for which predictions should be made.
