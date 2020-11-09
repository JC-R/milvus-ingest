#!/usr/bin/env bash

COLLECTION=$1
MAPFILE=$2
INPUTDIR=$3

# t5-base model
DIMS=768

for f in $INPUTDIR/*.npz; do
  f1=`echo $f | rev | cut -d "/" -f 1 | rev | sed -e "s/\.npz//g"`;
  segment=`echo $f1 | rev | cut -d "." -f 1 | rev`;
  fname=`echo $f1 | sed -e "s/[0-9.]*$//g"`;
  echo $f
done

python milvus-ingest.py -i $f -c $COLLECTION -d $DIMS -o $MAPFILE --test --append

