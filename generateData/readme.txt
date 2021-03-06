Writen by Karbo during the summer of 2017 and modified by Braden in the spring of 2019.

This is the readme for using the scripts inside generateData. 
The file is divided into three sections. The first section, SETUP, describes how to download and configure SUSYHIT and PROSPINO. The second section, RUN, describes the usage of makeData.sh, and the third section titled PROCESS describes how to use createDataSets.py.

SETUP
-----------------------------------------------------------
First, download SUSYHIT and PROSPINO. 

SUSYHIT: https://www.itp.kit.edu/~maggie/SUSY-HIT/

PROSPINO: http://www.thphys.uni-heidelberg.de/~plehn/index.php?show=prospino&visible=tools

Next, unpack SUSYHIT into a folder in the user's working directory named "susyhit" and PROSPINO into a folder name "prospino" also in the user's working directory. Prospino is by default in a folder of its own named "on_the_web_10_17_14". The contents of this folder should be unpacked into the "prospino" directory so that the path is ~/prospino/files instead of ~/prospino/on_the_web_10_17_14/files.

In order to actually run these programs be sure to execute make into the command line when in the directory for both of these programs.

Next, make sure GNU parallel is installed. If it is not type the command:
sudo apt-get install parallel
The number of cores used in parallel is a command line argument.

Finally, ensure that makeData.sh, datagroup.py, pointchange.py, and heading.txt are in the same working directory. All of the code, text, as well as the susyhit and prospino folders, should now be in the user's working directory.

The program will now run and output data into the working directory. The program will not, however, output the values needed for pMSSM in its default state.

To configure the program to work in pMSSM one must open ~/susyhit/suspect2_lha.in and make some changes. Specifically, the first line of code (which has the comment "mSUGRA") should have its second value changed from 1 to 0. As the list above this line states, 0 tells the program to use MSSM. 

Next one should comment out the lines in the MINPAR block and uncomment all of the lines in the EXTPAR block except for those numbered 14 15 and 16 (A_u A_d and A_e.)

Finally, one should open ~/prospino/prospino_main.f90 and change the collider energy to the approptiate value. The default is 14 TeV. This line is found in the third value block. 

RUN
----------------------------------------------------------
The program should now be ready to execute. It should be executed from the directory in which it and the two python scripts are located.

A WARNING: This prgram creates AND THEN DELETES temporary directories at its location named "susy" and "pros". If the user has directories in the script's directory which have these names THEY WILL BE DELETED when the script finishes executing.

Usage: makeData.sh [FLAGS] [NAME] [NUMBER] [CORES]

     Name     The name of the datafile to be created
     Number   An integer value of points to be generated
     Cores    An integer value of cores to be used

     Flags(optional):
       -h     Display help message
       -a     Append data to an existing datafile
       
PROCESS
----------------------------------------------------------
The script createDataSets.py processes the data output from makeData.sh to remove NaNs and create training files.

Usage: python createDataSets.py [readName] [writeName] [trainPercent]
    * readName: the name of the file to be read with file ending
    * writeName: the prefix of the written files, with no ending
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
use one, but not both, of these columns.
