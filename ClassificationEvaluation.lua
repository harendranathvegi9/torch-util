local ClassificationEvaluation = torch.class('ClassificationEvaluation')

--todo: make version of this for 

function ClassificationEvaluation:__init(batcher,net)
	self.batcher = batcher
	self.net = net
end

function ClassificationEvaluation:evaluate(epochNum)
	local count = 0
	local total_correct = 0
	self.batcher:reset()
	print('STARTING EVALUATION')
	self.net:evaluate()
	while(true) do
		local batch_labels, batch_inputs, num_actual_data = self.batcher:getBatch()
		if(batch_inputs == nil) then break end
		local preds = self.net:forward(batch_inputs)
		local _,pi=torch.max(preds,2)
		pi:narrow(1,1,num_actual_data)
		pi = pi:type(batch_labels:type())

		batch_labels:narrow(1,1,num_actual_data)
		local correct = pi:eq(batch_labels):sum()
		total_correct = total_correct + correct
		count = count + preds:size(1)
	end
	self.net:training()
	local acc = 100*total_correct/count

	print('Accuracy%: '..acc)
	print('computed on '..count.." examples")
	print('')
end

