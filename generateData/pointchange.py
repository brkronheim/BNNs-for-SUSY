"""pointchange.py 

Written by Karbo in the summer of 2017.

This code randomly assings each susy folder a different point in the pMSSM
parameter space.
The program takes the following two inputs:
    * The name of the datafield to be written 
    * The number of points to be generated. 

This code is made to work with the shell script makeData.sh. See the readme
for more information.

"""
import sys
import numpy
import random
import linecache

def main():
    with open(str(sys.argv[1]), 'r') as file:
        contents = file.readlines()

    value = list()
    value.append(str(numpy.random.uniform(-3,3))+'E+03') #M_1
    value.append(str(random.choice([numpy.random.uniform(-3,-0.1), numpy.random.uniform(0.1,3)]))+'E03') #M_2    ERROR
    value.append(str(numpy.random.uniform(0,3))+'E+03')  #M_3
    value.append(str(numpy.random.uniform(-7,7))+'E+03') #A_t
    value.append(str(numpy.random.uniform(-7,7))+'E+03') #A_b
    value.append(str(numpy.random.uniform(-7,7))+'E+03') #A_tau
    value.append('#') #A_u
    value.append('#') #A_d
    value.append('#') #A_e
    value.append(str(random.choice([numpy.random.uniform(-3,-0.1), numpy.random.uniform(0.1,3)]))+'E+03') #mu    ERROR
    value.append(str(numpy.random.uniform(0,3))+'E+03') #MA_pole
    value.append(str(numpy.random.uniform(0.2,6))+'E+01') #tanbeta
    value.append(str(numpy.random.uniform(0,3))+'E+03')  #M_eL
    value.append(value[12])                  #M_muL
    value.append(str(numpy.random.uniform(0,3))+'E+03') #M_tauL
    value.append(str(numpy.random.uniform(0,3))+'E+03') #M_eR
    value.append(value[15])                 #M_muR
    value.append(str(numpy.random.uniform(0,3))+'E+03') #M_tauR
    value.append(str(numpy.random.uniform(0.1,3))+'E+03') #M_q1L   ERROR
    value.append(value[18])                 #M_q2L
    value.append(str(numpy.random.uniform(0,3))+'E+03') #M_q3L
    value.append(str(numpy.random.uniform(0.1,3))+'E+03') #M_uR    ERROR
    value.append(value[21])                 #M_cR
    value.append(str(numpy.random.uniform(0,3))+'E+03') #M_tR
    value.append(str(numpy.random.uniform(0,3))+'E+03') #M_dR
    value.append(value[24])                 #M_sR
    value.append(str(numpy.random.uniform(0,3))+'E+03') #M_bR
    for i in range(27):
        if not (i == 6 or i == 7 or i == 8):
            row = contents[i+69]
            contents[i+69] = '          ' + row.split()[0] + '     ' + str(value[i]) + '   ' + row.split()[2] + ' ' + row.split()[3] + '\n'

    with open(str(sys.argv[1]), 'w') as file:
        file.writelines(contents)


main()
