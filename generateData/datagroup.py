"""datagroup.py

Written by Karbo in the summer of 2017 and modified by Braden in the spring of 2019

This code reads the data output of the individual susyhit and prospino datafiles
and writes them into one document. The program takes the following three arguments:
    * the name of the datafile to create
    * the number of the datafile to read
    * whether or or not the file is in setup phase
This version is called after every data point so the data is written incrementally.

This code is made to function with the shell script makeData.sh, see the readme 
for more information.
"""

import sys
import linecache

def main():
    name = sys.argv[1] #name of datafile to be used
    num = int(sys.argv[2]) #which file to read
    setup = sys.argv[3] #setup
    if(setup=="True"):
        outfile = open('../'+str(name), 'w') #create datafile
        row = linecache.getline('./heading.txt',1)
        outfile.write(row)
        outfile.close()
        
    else:
        try: #try and open the file
            outfile = open('../'+str(name), 'a')
            
        except FileNotFoundError: #the file was not initially setup, so do it here
            outfile = open('../'+str(name), 'w') #create datafile
            #get input names from a sus file and write to datafile
            for j in range(24):
                if not (j == 10 or j == 13 or j == 16 or j == 19 or j == 22):
                    row = linecache.getline('../susy/sus'+str(num)+'.dat', j+64)
                    outfile.write("{0:>16}".format(row.split()[3]))
        
            #get target names from a pro file and write to datafile
            row = linecache.getline('./pro'+str(num)+'.dat', 3)
            outfile.write("{0:>14}".format(row.split()[13]))
            outfile.write("{0:>14}".format(row.split()[14]))
            outfile.write("\n")
        
        #Combine the data from the sus and pro files
        row = linecache.getline('./pro'+str(num)+'.dat',1)
        if not (float(row.split()[14]) == float(0.610)):
            #read and write the sus data
            for j in range(24):
                if not (j == 10 or j == 13 or j == 16 or j == 19 or j == 22):
                    row = linecache.getline('../susy/sus'+str(num)+'.dat', j+64)
                    outfile.write("{0:>16}".format(row.split()[1]))
                
            #read and write the pro data
            row = linecache.getline('./pro'+str(num)+'.dat', 1) 
            outfile.write("{0:>14}".format(row.split()[14]))
            outfile.write("{0:>14}".format(row.split()[15]))
            outfile.write("\n")
        outfile.close()
        
main()
