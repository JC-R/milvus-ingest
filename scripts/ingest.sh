#!/usr/bin/env bash

FILES=$1
COLLECTION=$2
INPUTDIR=$3
OUTDIR=$4

# t5-base model
DIMS=768
echo $FILES
i=0
for f in $INPUTDIR/$FILES*.npz; do
  f1=`echo $f | rev | cut -d "/" -f 1 | rev | sed -e "s/\.npz//g"`;
  segment=`echo $f1 | rev | cut -d "." -f 1 | rev`;
  if [ $i -gt 0 ]; then
    APPEND=--append
  fi
  i=$((i+1))
  python milvus-ingest.py -i $f -c $COLLECTION -d $DIMS -o $OUTDIR/$FILES $APPEND
done



