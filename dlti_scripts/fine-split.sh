#!/bin/bash

# parse arguments, from: https://stackoverflow.com/a/29754866

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=cfm:s:g:
LONGOPTS=comments,fun_as_ret,mode:,ctx_size:,granularity:

# -use ! and PIPESTATUS to get exit code with errexit set
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

MODE=none
F=""
C=""
CTX_SIZE=0
SUFFIX=""
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -f|--fun_as_ret)
            F=-f
            shift
            ;;
        -c|--comments)
            C=-c
            shift
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -s|--ctx_size)
            CTX_SIZE="$2"
            shift 2
            ;;
        -g|--granularity)
            SUFFIX="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 4
            ;;
    esac
done

# handle non-option arguments
if [[ $# == 1 && $MODE == none ]]
then
    MODE=$1
elif [[ $# == 0 ]]
then
     if [[ $MODE == none ]]
     then
         echo "$0: Mode is required (as option or non-option argument)."
         exit 5
     fi
else
    echo "$0: Too many arguments"
    exit 6
fi

# nice funciton calls
function callf(){
    echo In $PWD:
    echo $@
    $@
}

# program proper
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPTDIR
source ../setdirs.sh

if [[ $MODE == context || $MODE == occurence ]]
then
    NAME=$MODE-$CTX_SIZE$F$C
else
    NAME=$MODE$F
fi

if [[ $SUFFIX ]]
then
    DATASET_NAME=$NAME-$SUFFIX
    SPLIT_NAME=$LOGDIR/$SUFFIX-split.txt
else
    DATASET_NAME=$NAME
    SPLIT_NAME=$LOGDIR/data-split.txt
fi

DATASET_DIR=$DATARELDIR/sets/$DATASET_NAME

cd $PROJDIR
if [ -d $DATASET_DIR ]; then rm -r $DATASET_DIR; fi
callf python3 $SCRIPTRELDIR/split_data.py $DATARELDIR/raw/$NAME $DATASET_DIR -l $SPLIT_NAME -r
cd $DATASET_DIR
callf python3 $SCRIPTDIR/create_vocab_and_generalise.py train.csv -a vocab.csv

for i in $(ls *.csv)
do
    if [[ $i != *"-general.csv" && $i != "vocab.csv" ]]
    then
        mv $i ${i%.csv}-original.csv
    fi
done
for i in $(ls *-general.csv)
do
    mv $i ${i%-general.csv}.csv
done
