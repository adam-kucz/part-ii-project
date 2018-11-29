#!/bin/bash
SRC_DIR="data/repos"
OUT_DIR="data/annotations"

for file in $(find $SRC_DIR -name "*.py")
do
    if [[ $(basename $file) != "__init__.py" ]]
    then
        target=${file#"$SRC_DIR"}
        python3 ml-type-inference/process_python_input.py "$file" "$OUT_DIR/${target%.py}.csv"
    fi
done
