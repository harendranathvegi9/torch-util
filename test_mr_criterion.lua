require 'torch'
require 'nn'
require 'optim'
require 'rnn'
require 'os'
require 'MapReduce'

require 'MinibatcherFromFile'
require 'MinibatcherFromFileList'
require 'MyOptimizer'
require 'OptimizerCallback'
require 'OnePassMiniBatcherFromFileList'
require 'ClassificationEvaluation'
require 'TaggingEvaluation'
require 'Util'
require 'FeatureEmbedding'
require 'MyReshape'
require 'cunn'



trainList = 'data/train1.list'
minibatch = 32
useCuda = false
lazyCuda = true
numRowsToGPU = 20


local batcher = MinibatcherFromFileList(trainList,minibatch,useCuda,preprocess,true,lazyCuda,numRowsToGPU)


local vocabSize = 35697
local embeddingDim = 200
local rnnHidSize = 50

local lstm = nn.Sequencer(nn.FastLSTM(embeddingDim, rnnHidSize))
local mapper = nn.Sequential():add(nn.Sequential():add(nn.LookupTable(vocabSize,embeddingDim)):add(nn.SplitTable(2)):add(lstm):add(nn.SelectTable(-1)):add(nn.Linear(rnnHidSize,2))):add(nn.SoftMax())

local reducer = nn.Max(2)
local mapReduce = nn.MapReduce(mapper,reducer)
mapReduce = mapReduce:cuda()
local labs,inputs,count = batcher:getBatch()

local out = mapReduce:forward(inputs)


out1 = out:split(1,2)

--print(out1[1][1])
local criterion = nn.MarginRankingCriterion(0.3):cuda()
df_do_tensor = torch.Tensor(out:size())
for i=1,labs:size(1) do 
	local err = criterion:forward({out1[1][i],out1[2][i]},labs[i])
	local df_do = criterion:backward({out1[1][i],out1[2][i]}, labs[i])
	-- print('labs[i]')
	-- print(labs[i])
	-- print('df_do')
	-- print(df_do)
	df_do_tensor[i][1] = df_do[1][1]
	df_do_tensor[i][2] = df_do[2][1]
end
print(df_do_tensor:cuda())
mapReduce:backward(inputs,df_do_tensor)

-- local  criterion = nn.ClassNLLCriterion():cuda()
-- local  err = criterion:forward(out,labs)
-- local df_do = criterion:backward(out,labs)
-- print('labs')
-- print(labs)
-- print('df_do')
-- print(df_do)



