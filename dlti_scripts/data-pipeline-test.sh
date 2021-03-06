#!/bin/bash

# parse arguments, from: https://stackoverflow.com/a/29754866

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=cfqm:s:l:g:
LONGOPTS=comments,fun_as_ret,fail_fast,mode:,ctx_size:,logdir:,granularity:

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
Q=""
ERROR_LOGDIR=""
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
        -q|--fail_fast)
            Q=--fail_fast
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
        -l|--logdir)
            ERROR_LOGDIR="$2"
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

if [[ $MODE == context || $MODE == occurence ]]
then
    NAME=$MODE-$CTX_SIZE$F$C
else
    NAME=$MODE$F
fi
NAME=$NAME-test

if [[ $ERROR_LOGDIR == "" ]]
then
    ERROR_LOGDIR=$NAME
fi

cd $PROJDIR
callf python3 $SCRIPTRELDIR/extract_data.py $MODE $DATARELDIR/repos-test $DATARELDIR/raw/$NAME --ctx_size $CTX_SIZE $F $C $Q --logdir $LOGDIR/$ERROR_LOGDIR

if [[ $MODE == context || $MODE == occurence ]]
then
    VARARGS="$C -s $CTX_SIZE"
else
    VARARGS=""
fi

$SCRIPTRELDIR/fine-split-test.sh $MODE $F -g "$SUFFIX" $VARARGS
