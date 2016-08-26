#!/bin/bash
#############################################
# run me with the following command to write a log file
# ./run.sh 2>&1 | tee run.log

# THIS BASH SCRIPT RUNS SINGLE EXPERIMENTS ON A SINGLE DATASET
# IMPORTANT: ALL FILES ARE IDENTIFIED BY THEIR NAME, SO MAKE SURE, THE NAMES ARE PRESERVED IN THE EXPERIMENT AND EVALUATION!!
#############################################

DATASET_NAME="GRAZ";
#DATASET_NAME="ICPR";

#METHOD_NAME="class";
METHOD_NAME="regr";

#FOREST_EXE="../../code/bin/celldetection-class/build/icgrf-celldet-class";
FOREST_EXE="../../code/bin/celldetection-regr/build/icgrf-celldet-regr";


echo "Running experiment from $(pwd)";

# get the ID of the experiment
expID=$(basename $(pwd));
echo "******************************"
echo "Executing experiment '$expID'"
echo "******************************"
	
# get the config.txt in that directory
CONFIG_FILE="config.txt";
PATH_BINDATA="./bindata";
PATH_TREES="$PATH_BINDATA/trees";
PATH_PREDICTIONS="$PATH_BINDATA/predictions";
mkdir $PATH_BINDATA;
mkdir $PATH_TREES;
mkdir $PATH_PREDICTIONS;
LOG_FILE="$PATH_BINDATA/${DATASET_NAME}_${METHOD_NAME}.log";

echo "=============== FOREST START ==============="
echo "Training forest and predicting test images";

# execute the forest and redirect the output to the experiment's log file
$FOREST_EXE $CONFIG_FILE > $LOG_FILE;

echo "=============== FOREST END ==============="

echo "Running the post-processing script and computing performance data..."
echo "=============== MATLAB START ==============="
# run the matlab script
EVAL_SCRIPT="evaluate"; #OR evaluate.m
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
	echo "platform = Linux";
	matlab -nodesktop -nodisplay -nosplash -r "run('$EVAL_SCRIPT'); exit;";
elif [[ "$unamestr" == 'Darwin' ]]; then
	echo "platform = Mac OS X";
	/Applications/MATLAB_R2013a.app/bin/matlab -nodesktop -nodisplay -nosplash -r "run('$EVAL_SCRIPT'); exit;";
else
	echo "ERROR: Cannot run matlab post-processing!";
fi
echo "=============== MATLAB END ==============="
printf "Finished experiment '$expID'.\n\n"
