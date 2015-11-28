require 'LabeledDataFromFile'
require 'SparseMinibatcherFromFile'
require 'os'
local MinibatcherFromFile = torch.class('MinibatcherFromFile')


function MinibatcherFromFile:__init(file,batchSize,cuda,shuffle,lazyCuda,numRowsToGPU)

	if(lazyCuda) then assert(numRowsToGPU > 0) end
	self.batchSize = batchSize
	self.doShuffle = shuffle
	self.numRowsToGPU = numRowsToGPU
	self.lazyCuda = lazyCuda

	print('reading from '..file)
	local loadedData = torch.load(file)
	if(loadedData.isSparse) then 
		self.isSparse = true
		self.sparseBatcher = SparseMinibatcherFromFile(loadedData,batchSize,cuda,shuffle) 
	else

		local loaded = LabeledDataFromFile(loadedData,cuda,batchSize,lazyCuda)
		self.unpadded_len = loaded.unpadded_len
		assert(self.unpadded_len ~= nil)

		if(cuda) then
			self.labels = loaded.labels_pad:cuda()
			self.data = loaded.inputs_pad:cuda()
		else
			self.labels = loaded.labels_pad
			self.data = loaded.inputs_pad
			assert(self.labels:size(1) == self.data:size(1))
			if(lazyCuda) then 				
				local sizes = self.data:size()
				local label_sizes = self.labels:size()

				numRowsToGPU = math.min(numRowsToGPU, sizes[1])				
				--allocate a tensor and push it to gpu				
				sizes[1] = numRowsToGPU
				label_sizes[1] = numRowsToGPU
				self.gpuData = torch.Tensor(torch.LongStorage(sizes)):cuda()
				self.gpuLabels = torch.Tensor(torch.LongStorage(label_sizes)):cuda()
				self.gpuData:copy(self.data:narrow(1,1,numRowsToGPU))
				self.gpuLabels:copy(self.labels:narrow(1,1,numRowsToGPU))
				self.numRowsToGPU = numRowsToGPU -- this is the max size of the gpu tensor
				self.endGPUData = numRowsToGPU --endGPUData points to the last location of the gpu tensor. This might not neccessarily be the end of the gpu tensor, since the last shard of the data might be smaller in length of the tensor
				self.curStartGPU = numRowsToGPU + 1 --currStartGPU points to the place in data from which we will start copying to GPU.
				if(self.curStartGPU > self.data:size()[1]) then self.curStartGPU = 1 end

			end
		end
		assert(self.labels:size(1) == self.data:size(1))
		self.numRowsValue = self.data:size(1)
		self.curStart = 1
		self.curStartSequential = 1		
	end
end


function MinibatcherFromFile:copyNextRowsToGPU()

	local numRowsToGPU = self.numRowsToGPU
	local startIdx = self.curStartGPU
	local endIdx = startIdx + numRowsToGPU - 1
	if(endIdx > self.unpadded_len) then
		numRowsToGPU = self.unpadded_len - startIdx + 1 
	end
	self.gpuData:narrow(1,1,numRowsToGPU):copy(self.data:narrow(1,startIdx,numRowsToGPU))	
	self.gpuLabels:narrow(1,1,numRowsToGPU):copy(self.labels:narrow(1,startIdx,numRowsToGPU))
	self.endGPUData = numRowsToGPU --endGPUData points to the last location of the gpu tensor. This might not neccessarily be the end of the gpu tensor, since the last shard of the data might be smaller than length of the tensor
	self.curStartGPU = startIdx + numRowsToGPU
	if(self.curStartGPU > self.unpadded_len) then 
		self.curStartGPU = 1
		self:shuffle()
	end	

end

function MinibatcherFromFile:numRows()
	if(self.isSparse) then return self.sparseBatcher.numRows end
	return self.numRowsValue
end

function MinibatcherFromFile:shuffle()
	if(self.isSparse) then return self.sparseBatcher:shuffle() end
	if(self.doShuffle) then
		 local inds = torch.randperm(self.labels:size(1)):long()
		 self.labels = self.labels:index(1,inds)
		 self.data = self.data:index(1,inds)
		 self.curStart = 1
		 self.curStartSequential = 1
	end
end

function  MinibatcherFromFile:getBatch()
	if(self.isSparse) then return self.sparseBatcher:getBatch() end

	if(self.lazyCuda and self.curStart > self.endGPUData) then
		self.curStart = 1
		--copy next shard of data to GPU
		self:copyNextRowsToGPU()
	end
	local startIdx = self.curStart
	local endIdx = startIdx + self.batchSize-1
	if(self.lazyCuda) then 
		endIdx = math.min(endIdx,self.endGPUData)
	else 
		endIdx = math.min(endIdx,self.numRowsValue)
	end

	self.curStart = endIdx + 1
	if(not self.lazyCuda and self.curStart > self.unpadded_len) then
		self.curStart = 1
		self:shuffle()
	end
	if(self.lazyCuda) then		
		local batch_labels = self.gpuLabels:narrow(1,startIdx,endIdx-startIdx+1)
		local batch_data = self.gpuData:narrow(1,startIdx,endIdx-startIdx+1)
		local num_actual_data = endIdx-startIdx+1 --no padding
		return batch_labels,batch_data, num_actual_data	
	else
		local batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
		local batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)
		local num_actual_data = self.batchSize
		if(endIdx > self.unpadded_len) then
			num_actual_data = self.unpadded_len - startIdx +1 
		end		
		return batch_labels,batch_data, num_actual_data	
	end
	
end

function MinibatcherFromFile:reset()
	if(self.isSparse) then return self.sparseBatcher:reset() end

	self.curStartSequential = 1
	self.curStart = 1
end

function  MinibatcherFromFile:getBatchSequential()
	if(self.isSparse) then return self.sparseBatcher:getBatchSequential() end

	local startIdx = self.curStartSequential
	local endIdx = startIdx + self.batchSize-1

	endIdx = math.min(endIdx,self.numRowsValue)
	self.curStartSequential = endIdx +1
	if(startIdx > self.unpadded_len) then
		return nil
	end
	local num_actual_data = self.batchSize

	if(endIdx > self.unpadded_len) then
		endIdx = self.unpadded_len - (self.unpadded_len % 32)
		if(endIdx < self.unpadded_len) then endIdx = endIdx + 32 end
		self:shuffle()
	end
	num_actual_data = math.min(self.unpadded_len - startIdx,endIdx - startIdx) + 1


	local batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
	local batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)

	assert(num_actual_data <= batch_labels:size(1))

	if(self.lazyCuda) then
		batch_labels = batch_labels:cuda()
		batch_data = batch_data:cuda()
	end

	return batch_labels,batch_data, num_actual_data
end
