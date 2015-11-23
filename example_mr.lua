require 'nn'
require 'MapReduce'
require 'rnn'

local vocabSize = 35697
local embeddingDim = 200
local rnnHidSize = 50
local data = torch.rand(26,29,16):mul(vocabSize):ceil()


local lstm = nn.Sequencer(nn.FastLSTM(embeddingDim, rnnHidSize))
local mapper = nn.Sequential():add(nn.Sequential():add(nn.LookupTable(vocabSize,embeddingDim)):add(nn.SplitTable(2)):add(lstm):add(nn.SelectTable(-1)):add(nn.Linear(rnnHidSize,2))):add(nn.LogSoftMax())

local reducer = nn.Max(2)
local mapReduce = nn.MapReduce(mapper,reducer)
--
local out = mapReduce:forward(data)
--
local go = out:clone():fill(1.0)
--
