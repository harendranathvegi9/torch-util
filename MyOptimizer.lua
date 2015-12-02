local MyOptimizer = torch.class('MyOptimizer')


--NOTE: various bits of this code were inspired by fbnn Optim.lua 3/5/2015


function MyOptimizer:__init(model,modules_to_update,criterion, trainingOptions,optInfo,isMarginCriterion)
    local model_utils = require 'model_utils'

     assert(trainingOptions)
	 assert(optInfo)
     self.structured = structured or false
	 self.model = model
	 self.optState = optInfo.optState	
     self.optConfig = optInfo.optConfig
     self.optimMethod = optInfo.optimMethod
     self.regularization = optInfo.regularization
     self.trainingOptions = trainingOptions 
     self.totalError = torch.Tensor(1):zero()
     self.checkForConvergence = optInfo.converged ~= nil
     self.optInfo = optInfo
     self.minibatchsize = trainingOptions.minibatchsize
     self.isMarginCriterion = isMarginCriterion or false

    local parameters
    local gradParameters

    if(Util:isArray(modules_to_update)) then
            parameters, gradParameters = model_utils.combine_all_parameters(unpack(modules_to_update))
    else
            parameters, gradParameters = modules_to_update:getParameters()
    end
    self.parameters = parameters
    self.gradParameters = gradParameters

    self.l2s = {}
    self.params = {}
    self.grads = {}
    for i = 1,#self.regularization.params do
            local params,grad = self.regularization.params[i]:parameters()
            local l2 = self.regularization.l2[i]
            table.insert(self.params,params)
            table.insert(self.grads,grad)
            table.insert(self.l2s,l2)
    end
    self.numRegularizers = #self.l2s


    self.cuda = optInfo.cuda
     if(optInfo.useCuda) then
        self.totalError:cuda()
    end

     self.criterion = criterion
    for hookIdx = 1,#self.trainingOptions.epochHooks do
        local hook = self.trainingOptions.epochHooks[hookIdx]
        if( hook.epochHookFreq == 1) then
            hook.hook(0)
        end
    end

end

function MyOptimizer:train(batchSampler)
	 local prevTime = sys.clock()
     local batchesPerEpoch = self.trainingOptions.batchesPerEpoch
     --local tst_lab,tst_data = batchSampler()
     local epochSize = batchesPerEpoch*self.minibatchsize
     local numProcessed = 0
     
    local i = 1
    while i < self.trainingOptions.numEpochs and (not self.checkForConvergence or not self.optInfo.converged) do
        self.totalError:zero()
        for j = 1,batchesPerEpoch do
    	    local minibatch_targets,minibatch_inputs = batchSampler()
            if(minibatch_targets) then
                numProcessed = numProcessed + minibatch_targets:nElement() --this reports the number of 'training examples.' If doing sequence tagging, it's the number of total timesteps, not the number of sequences. 
            else
                --in some cases, the targets are actually part of the inputs with some weird table structure. Need to account for this.
                numProcessed = numProcessed + self.minibatchsize
            end
            self:trainBatch(minibatch_inputs,minibatch_targets)
        end
        local avgError = self.totalError[1]/batchesPerEpoch
        local currTime = sys.clock()
        local ElapsedTime = currTime - prevTime
        local rate = numProcessed/ElapsedTime
        numProcessed = 0
        prevTime = currTime
        print(string.format('\nIter: %d\navg loss in epoch = %f\ntotal elapsed = %f\ntime per batch = %f',i,avgError, ElapsedTime,ElapsedTime/batchesPerEpoch))
        --print(string.format('cur learning rate = %f',self.optConfig.learningRate))
        print(string.format('examples/sec = %f',rate))
        self:postEpoch()

         for hookIdx = 1,#self.trainingOptions.epochHooks do
            local hook = self.trainingOptions.epochHooks[hookIdx]
	        if( i % hook.epochHookFreq == 0) then
                hook.hook(i)
            end
	   end
       i = i + 1
    end
end

function MyOptimizer:postEpoch()
    --this is to be overriden by children of MyOptimizer
end

function MyOptimizer:trainBatch(inputs, targets)

    --print('Starting to train a batch')
    assert(inputs)
    assert(targets)
    for i=1,targets:size()[1] do targets[i] = targets[i] %2 end
    print(targets)
    local parameters = self.parameters
    local gradParameters = self.gradParameters
    local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        self.model:zeroGradParameters()
        local output = self.model:forward(inputs)
        local df_do = nil
        local err = nil
        if(self.isMarginCriterion) then
            local out1 = output:split(1,2)
            df_do_tensor = torch.Tensor(output:size())
            err = 0
            for i=1,targets:size(1) do 
                local y = 1
                if(targets[i] == 1) then y = 1 else y = -1 end
                err = err + self.criterion:forward({out1[1][i],out1[2][i]},y)
                local df_do = self.criterion:backward({out1[1][i],out1[2][i]},y)
                df_do_tensor[i][1] = df_do[1][1]
                df_do_tensor[i][2] = df_do[2][1]
            end
            df_do = df_do_tensor:cuda()
        else
            err = self.criterion:forward(output, targets)
            df_do = self.criterion:backward(output, targets)
        end        
        self.model:backward(inputs, df_do)
        -- self.model:updateGradParameters(0.9)
        -- self.model:updateParameters(0.01)
        --note we don't bother adding regularizer to the objective calculation. who selects models on the objective anyway?
        for i = 1,self.numRegularizers do
            local l2 = self.l2s[i]
            for j = 1,#self.params[i] do
                self.grads[i][j]:add(l2,self.params[i][j])
            end
        end
        self.totalError[1] = self.totalError[1] + err
    	gradParameters:div(self.minibatchsize)
    	--self.model:gradParamClip(5)
        gradParameters:clamp(-5, 5)
    	return err, gradParameters
    end

    self.optimMethod(fEval, parameters, self.optConfig, self.optState)
    return err
end

