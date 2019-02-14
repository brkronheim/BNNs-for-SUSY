import click
import subprocess
import sys

import pandas as pd

@click.command()
@click.option("--file", default="experiments.txt", help="Name of file with network descriptions")

def main(file):
    """Runs the experiments detailed in --file.
    
    This file should be formated as:
    
    depth width epochs
    
    with a space between each number and a new line for each experiment.
    """
    experiments=pd.read_csv(file, sep = " ", header = None)
    shape=experiments.shape
    for row in range(shape[0]):
        depth = str(experiments.iat[row, 0])
        width = str(experiments.iat[row, 1])
        epochs = str(experiments.iat[row, 2])
        
        name = width + "Wide_" + depth + "Deep_" + epochs + "Epochs"
        print("Running Network", name)
        
        #call the command line command
        depth = "--hidden=" + depth
        width = "--width=" + width
        epochs = "--epochs=" + epochs
        tb = "--tb=" +name
        subprocess.call(["python3","BNN_train.py", depth, width, epochs, tb])
          
            
if (__name__=="__main__"):
    main()
