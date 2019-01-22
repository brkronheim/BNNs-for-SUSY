#!/bin/bash

#------------------------------------------------------------------------
#makeData.sh, a code written by Karbo in the summer of 2017 and reworked 
#for parallel processing by Braden in the spring of 2019
#This code generates pMSSM datapoints using the programs susyhit and prospino. 
#The program takes three arguments. First the name of the datafile to be written,
#second the number of points to be generated, and third the number of cores to
#be used
#This file relies on pointchange.py and datagroup.py. 
#View Readme.txt for more info.
#------------------------------------------------------------------------

a=0

usage="$(basename "$0") [Flags] [Name] [Number] [Cores] -- Program to generate pMSSM predictions

    Name    The name of the datafile to be created
    Number  An integer value of points to be generated
    Cores   An integer value of cores to be used

    Flags(optional):
      -h    Display this message
      -a    Append data to an existing datafile"


while getopts ':ha:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    a) a=1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
shift "$((OPTIND - 3))"
done


name=$1 #first argument is the name of the file to right to
number=$2 #second argument is the number of points to generate
cores=$3 #thrid argument is the number of cores to use
percent=$number/100

gensusy () {
	i=$1 
	number=$2

	#run susyhit and copy output
	cd ./point$i/ 
	./run > /dev/null
	cd ..
	cp ./point$i/susyhit_slha.out ./sus$i.dat
	mv ./point$i/susyhit_slha.out ./../pros/point$i/prospino.in.les_houches
	rm -R ./point$i
}

genprospino () {
	i=$1 
	number=$2
	name=$3
		
        #run prospino and copy output output
	cd ./point$i/
	./prospino_2.run > /dev/null
	cd ..
	cp ./point$i/prospino.dat ./pro$i.dat
	rm -R ./point$i
}

main () {
	i=1
	if [ "$a" = "1" ]
	then
	    i=2
	fi
	while [ "$i" -le "$number" ];
	do
	    #make directories and copy files
	    mkdir -p ./susy/point$i
	    mkdir -p ./pros/point$i
	    cp ./susyhit/run ./susy/point$i
	    cp ./susyhit/suspect2_lha.in ./susy/point$i
	    cp ./susyhit/susyhit.in ./susy/point$i
	    cp -R ./prospino/* ./pros/point$i
	    
	    #change the inputs for susyhit
	    python pointchange.py ./susy/point$i/suspect2_lha.in
	    
	    #Monitor the percentage of work completed
	    if [ $(($i%$percent)) -eq 0 ]
	    then
		echo $((100*$i/$number)) percent of points created
	    fi
		
	    i=$(($i+1))
	done

	echo All Points Created

	cd ./susy

	i=1
	if [ "$a" = "1" ]
	then
	    i=2
	fi

        export -f gensusy

	parallel --link --bar -j$cores gensusy ::: $(seq $i $number) ::: $number

	cd ..

	echo All Susyhit Calculations Complete

	cd ./pros


	i=1
	    
	#run prospino and copy output

        export -f genprospino

        parallel --link --bar -j$cores genprospino ::: $(seq $i $number) ::: $number ::: $name ::: 0
	python ../datagroup.py $name $i "True"

	for n in $(seq $i $number); do
	    python ../datagroup.py $name $n "False"
	done

	cd ..

	echo All Prospino Calculations Complete

	rm -R ./susy
	rm -R ./pros

	echo Temporary Files Cleaned and Output Written to $name
}

main
