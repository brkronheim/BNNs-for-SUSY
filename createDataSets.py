"""processData.py

Written by Braden Kronheim in the spring of 2019.

Processes data output from makeData.sh to remove NaNs and create training files.
Takes the following three inputs:
    * readName: the name of the file to be read with file ending
    * write Name: the prefix of the written files, with no ending
    * trainPercent: the percent (0-100) as an integer of the 
    *               input data to be used for training data,
    *               the rest will be used for validation

Based upon the previous inputs, a certain amount of the data will be
written to the files ending in "TrainInput.txt", "TrainOutput.txt", 
"ValidateInput.txt", and "ValidateOutput.txt".
The Train files are to be used to train a nerual network, the Validate
files are to be used to validate the neural network, in Input files are
the input the networks take, and the ouput files are the output.

The output files will have two columns, one with the leading order calculations
and the other with the non leading order calculations. When training on this data
use one, but not both, of these columns
"""
import sys

def main():
    readName = sys.argv[1] #Name of file to be read
    writeName = sys.argv[2] #Prefix of written files
    trainPercent = int(sys.argv[3]) #Percent of data to be used for training
    
    with open(readName, "r") as inData:
        trainIn = open(writeName+"TrainInput.txt", "w")
        trainOut = open(writeName+"TrainOutput.txt", "w")
        validIn = open(writeName+"ValidateInput.txt", "w")
        validOut = open(writeName+"ValidateOutput.txt", "w")
        
        currentLine=0
        for line in inData:
            splitLine=line.split()
            if not (splitLine[20].lower() == "nan"): #Verifies the NLO terms are persent
                entry=0
                inWrite=""
                outWrite=""
                for entries in splitLine:
                    if(entry<19): #Input pMSSM parameters
                        inWrite += entries + "\t"
                    else: #Output cross sections
                        outWrite += entries + "\t"
                    entry+=1
                
                if(currentLine%100 < trainPercent): #Write to the train files   
                    trainIn.write(inWrite.strip()+"\n")
                    trainOut.write(outWrite.strip()+"\n")
                else: #Write to the validation files
                    validIn.write(inWrite.strip()+"\n")
                    validOut.write(outWrite.strip()+"\n")
            currentLine+=1

        trainIn.close()
        trainOut.close()
        validIn.close()
        validOut.close()

if __name__ == "__main__":
    main()
    