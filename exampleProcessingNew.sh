#!/bin/sh

outDir=data/train
dataset=train
int2torch="th int2torchNew.lua"
rm -f $outDir/$dataset.list #the downstream training code reads in this list of filenames of the data files, split by length
echo converting $dataset to torch files
for ff in $outDir/*.int 
do
out=`echo $ff | sed 's|.int$||'`.torch
$int2torch -input $ff -output $out -tokenLabels 0 -tokenFeatures 0 -addOne 1 #convert to torch format
echo $out >> $outDir/$dataset.list
echo $out
done
