#!/bin/bash

# parse arguments, from: https://stackoverflow.com/a/29754866

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=fm:c:
LONGOPTS=fun_as_ret,mode:,ctx_len:

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

F=""
CTX_LEN=0
MODE=none
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -f|--fun_as_ret)
            F=-f
            shift
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -c|--ctx_len)
            CTX_LEN="$2"
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
source ../setdirs.sh

if [[ $MODE == context ]]
then
    NAME=$MODE-$CTX_LEN$F
else
    NAME=$MODE$F
fi

cd $PROJDIR
callf python3 $SCRIPTRELDIR/extract_data.py $MODE $DATARELDIR/repos $DATARELDIR/raw/$NAME -s $CTX_LEN $F
callf python3 $SCRIPTRELDIR/split_data.py $DATARELDIR/raw/$NAME $DATARELDIR/sets/$NAME -l $LOGDIR/data-split.txt -r
cd $DATARELDIR/sets/$NAME
callf python3 $SCRIPTDIR/create_vocab_and_generalise.py train.csv -g train.csv validate.csv test.csv

mv train.csv train-original.csv
mv validate.csv validate-original.csv
mv test.csv test-original.csv

mv train-general.csv train.csv
mv validate-general.csv validate.csv
mv test-general.csv test.csv
