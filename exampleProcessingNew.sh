#!/bin/bash
outDir="/home/rajarshi/torch-util/data"
data_set=("train" "dev" "test")
int2torch="th int2torchNew.lua"
for set in "${data_set[@]}"
do
	echo $set
	dataDir=${outDir}/${set}
	dataset=$set
	if [ -f ${outDir}/${dataset}.list ]; then
		rm ${outDir}/${dataset}.list #the downstream training code reads in this list of filenames of the data files, split by length
	fi
	echo "converting $dataset to torch files"	
	for ff in $dataDir/*.int 
	do	
		out=`echo $ff | sed 's|.int$||'`.torch
		$int2torch -input $ff -output $out -tokenLabels 0 -tokenFeatures 0 -addOne 1 #convert to torch format
		echo $out >>  ${outDir}/${dataset}.list		
		echo ${out}
	done
done