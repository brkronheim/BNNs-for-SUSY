## Usage
The scripts within this folder allow the easy creation of networks with normal distributions for weights and biases. While these networks do not perform as well as the Bayesian nerual networks, they do outperform normal neural networks and train at about the same rate as them. Thus, these serve as an easy upgrade to traditional neural networks.

The train.py script in this folder is an implementation specifically for the the supersymmetry dataset. Different network architectures for training on it can be implemented via the command line with the following flags:

*  --hidden INTEGER     Number of hidden layers
*  --width INTEGER      Width of the hidden layers
*  --epochs INTEGER     Number of epochs to train for
*  --burnin INTEGER     Number of burnin epochs
*  --increment INTEGER  Epochs between saving networks
*  --cores INTEGER      Number of cores which can be used
*  --name TEXT          Name of network
*  --help               Show this message and exit.

In order to change the training set replace lines 36-39 in train.py, shown below, with the desired data sets
```
    trainIn=np.loadtxt("fullTrainInput.txt",delimiter="\t",skiprows=1)
    trainOut=np.loadtxt("fullTrainOutput.txt",delimiter="\t",skiprows=1)
    valIn=np.loadtxt("fullValidateInput.txt",delimiter="\t",skiprows=0)
    valOut=np.loadtxt("fullValidateOutput.txt",delimiter="\t",skiprows=0)
```

Using the script BNN_load.py it is possible to load saved networks and make predictions from them. Once again, the default for the script is to use the sumpersymmetry dataset. This script calculates the mean percent error and the amount of data within, 1, 2, and 3 standard deviations of the predicgion distributions made by the networks. The following flags configure how the predictions are made:

* --name TEXT     Name of network to load
* --out TEXT      Name of file to which analysis is written
* --iters INTEGER Number of iterations over the input data

In order to change the data set change lines 28-31 in same fashion as above.
