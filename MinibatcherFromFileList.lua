require 'os'

local MinibatcherFromFileList = torch.class('MinibatcherFromFileList')

function MinibatcherFromFileList:__init(fileList,batchSize,cuda,lazyCuda,preprocess,shuffle,numBatchesToCache)
	self.shuffle = shuffle
	if(not preprocess) then
		preprocess = function(a,b,c) return a,b,c end
	end
	self.preprocess = preprocess
	self.batches = {}
	self.currentBatchIds = {}
	local counts = {}	
	self.debugMode = false
	print(string.format('reading file list from %s',fileList))

	for file in io.lines(fileList) do
		local batch  = MinibatcherFromFile(file,batchSize,cuda,lazyCuda,shuffle)
		table.insert(counts,batch:numRows())
		table.insert(self.batches,batch)
	end
	self.weights = torch.Tensor(counts)
	self.weights:div(torch.sum(self.weights))
		self.debug = nil
		self.debug2 = nil
		self.debug3 = nil
		self.called = false
	self.numBatchesToCache = numBatchesToCache
	self.lazyCuda = lazyCuda
	self.batchCounter = 0
end


--[[
Implentation of lazy loading to GPU. It puts $numBatchesToCache into GPU and getBatch returns the next batch from this set.
--]]
function  MinibatcherFromFileList:cacheBatches()
	print('Starting to cache next set of batches')
	--go over currentBatchIds and remove them from gpu
	if(lazyCuda and self.batchCounter > 0) then
		print ('Removing previous data from GPU')
		for _,idx in ipairs(self.currentBatchIds) do
			self.batches[idx]:float()
		end
	end
	--reset the ids
	self.currentBatchIds = {}
	self.batchCounter = 0
	--sample the batch indices and store them
	for i=1,self.numBatchesToCache do 
		local idx = self:getNextBatchId()
		table.insert(self.currentBatchIds,idx) 
		if(self.lazyCuda) then --call :cuda for these batches
			self.batches[idx]:cuda()
		end
	end
end


function  MinibatcherFromFileList:getBatch()
	if((self.batchCounter == 0) or (self.batchCounter == self.numBatchesToCache)) then
		--cache the next set of batches
		self:cacheBatches()
	end
	--print(self.currentBatchIds)
	batch_labels,batch_data, num_actual_data = self.preprocess(self.batches[self.currentBatchIds[self.batchCounter+1]]:getBatch())
	self.batchCounter = self.batchCounter + 1
	return batch_labels,batch_data, num_actual_data
end

function  MinibatcherFromFileList:getNextBatchId()
	if(self.debugMode) then
		-- if(self.called) then
		-- 	return self.debug, self.debug2, self.debug3
		-- else
		local idx = torch.multinomial(self.weights,1)			
		-- self.debug, self.debug2, self.debug3 = preprocess(self.batches[idx[1]]:getBatch())
		-- self.called = true

		-- return self.debug,self.debug2, self.debug3
		return idx[1]
		--end	
	end

	local idx = torch.multinomial(self.weights,1)
	return idx[1]
	--return self.preprocess(self.batches[idx[1]]:getBatch())
end

function MinibatcherFromFileList:getAllBatches()
	local t = {}
	if(self.debugMode) then 
		local x,y,z = self.preprocess(unpack(self.batches[1]:getBatch()))
		table.insert(t,{x,y,z})
	else	
		for _,b in ipairs(self.batches) do
			while(true) do
				local lab,data,unpadded_len = b:getBatchSequential()
				if(data == nil) then break end
				local a,b,c = self.preprocess(lab,data,unpadded_len)
				table.insert(t,{a,b,c})
			end
		end
	end
	return t
end
