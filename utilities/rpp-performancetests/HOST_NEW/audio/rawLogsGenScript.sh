#!/bin/bash

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

# Input AUDIO_FILES - Three AUDIO_FILES (224 x 224)
DEFAULT_SRC_FOLDER="$cwd/../../../TEST_AUDIO_FILES/single_channel/"

# Output AUDIO_FILES
mkdir "$cwd/../../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
DEFAULT_DST_FOLDER="$cwd/../../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"

# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER="$DEFAULT_SRC_FOLDER"
DST_FOLDER="$DEFAULT_DST_FOLDER"

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>
if [[ "$1" -lt 0 ]] | [[ "$1" -gt 2 ]]; then
    echo "The starting case# must be in the 0-2 range!"
    echo
    echo "The testAllScript.sh bash script runs the RPP audio unittest testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScript.sh <S> <E>"
    echo "S     CASE_START (Starting case# (0-2))"
    echo "E     CASE_END (Ending case# (0-2))"
    exit 1
fi

if [[ "$2" -lt 0 ]] | [[ "$2" -gt 2 ]]; then
    echo "The ending case# must be in the 0-2 range!"
    echo
    echo "The testAllScript.sh bash script runs the RPP audio unittest testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScript.sh <S> <E>"
    echo "S     CASE_START (Starting case# (0-2))"
    echo "E     CASE_END (Ending case# (0-2))"
    exit 1
fi

if (( "$#" < 2 )); then
    CASE_START="0"
    CASE_END="2"
else
    CASE_START="$1"
    CASE_END="$2"
fi

rm -rvf "$DST_FOLDER"/*
shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all Audio Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./Tensor_host_audio <src folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=2;bitDepth<3;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        printf "\n./Tensor_host_audio $SRC_FOLDER_1 $bitDepth $case "
        ./Tensor_host_audio "$SRC_FOLDER" "$bitDepth" "$case" | tee -a "$DST_FOLDER/Tensor_host_audio_host_raw_performance_log.txt"

        echo "------------------------------------------------------------------------------------------"
    done
done

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>