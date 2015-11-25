#!/bin/sh

#input data
d=./data #this is where the processed data is
tokFeats=0 #use whatever the prepocessing did
tokLabels=0



#optional initialization from a checkpoint of a previous training run (make sure the architecture is the same!)
initModel= #use this if you don't want to specify an init model
#initModel=expts/1//model-5

#optional use of pretrained word embeddings (this option not used if you specify initModel)
initEmbeddings= #use this if you don't want to use embeddings
#initEmbeddings=$d/embeddings


#output 
exptDir=../VSMKBC/expts/1/ #where all the models, etc will be put
log=$exptDir/log.txt #where everything will be logged
saveFrequency=500 #how often to checkpoint

#training options
architecture=rnn
rnnType=lstm
lr=0.1 #learning rate. TODO: make a more verbose framework for specifying optimization options on the command line
gpuid=0 #if >= 0, then do computation on GPU
minibatch=32 #if using gpu, minibatch sizes needs to be a multiple of 32.
lazyCuda=1
numBatchesToCache=30 #no of batches to be loaded to gpu if lazyCuda is true
#l2=0.00001
l2=0
#embeddingL2=0.00001
embeddingL2=0
convWidth=3 ##make sure this is compatible with the amount o=f padding used in exampleProcessing.sh. If  convWidth = 3, need pad = 1. If convWidth=5, need pad = 2...
##IMPORTANT: if using token features, you should also tweak the per-feature-template embedding sizes below (eg, tokenString:50)



################you shouldn't need to change anything below here ################
labelDim=`cat $d/domain.labelDomainSize.txt`
vocabSize=`cat $d/domain.domainSizes.txt  | grep '^tokenString' | cut -f2`

#see ModelTraining.lua for documentation of its command line options
dataOptions="$d/train.list -testList $d/train.list -tokenFeatures $tokFeats -tokenLabels $tokLabels -labelDim $labelDim -vocabSize $vocabSize"
options="-trainList $dataOptions -minibatch $minibatch -gpuid $gpuid -lazyCuda $lazyCuda -numBatchesToCache $numBatchesToCache  -learningRate $lr -l2 $l2 -embeddingL2 $embeddingL2"


if [ "$initEmbeddings" != "" ]; then
	options="$options -initEmbeddings $initEmbeddings"
fi 

if [ "$initModel" != "" ]; then
	options="$options -initModel $initModel"
fi 


if [ "$tokFeats" == "0" ]; then
	##user-specified embeddings sizes
	embeddingDim=200 ##important: if use initEmbeddings, the dimensionality specified here needs to be the same as the tensor in $initEmbeddings
	featureDim=15

	moreOptions="-embeddingDim $embeddingDim -featureDim $featureDim"
	options="$options $moreOptions"
else
	tmp=/tmp/domainSizes
	
	##user-specified embeddings sizes
	echo tokenString:50 > $tmp ##important: if use initEmbeddings, the dimensionality specified here needs to be the same as the tensor in $initEmbeddings
	echo isCap:5 >> $tmp
	echo isNumeric:3 >> $tmp


	cat $tmp | tr ':'  '\t' > $tmp.2
	python mergeMaps.py $d/domain.domainSizes.txt $tmp.2 > $tmp.merge

	moreOptions="-featureEmbeddingSpec $tmp.merge"
	options="$options $moreOptions"
fi

mkdir -p $exptDir

modelBase=$exptDir/model
options="$options -saveFrequency $saveFrequency -model $modelBase "
#Train a RNN shall we?
rnnType='lstm'
architecture='rnn'
options="$options -architecture $architecture -rnnType $rnnType"
cmd="th NewModelTraining.lua $options"
echo Executing:
echo $cmd
$cmd | tee $log


