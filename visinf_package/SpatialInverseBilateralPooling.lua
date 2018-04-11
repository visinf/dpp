--  Copyright (c) 2018, TU Darmstadt.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree.


local C = visinf.C

local SpatialInverseBilateralPooling, parent = torch.class('visinf.SpatialInverseBilateralPooling', 'nn.Module')


function SpatialInverseBilateralPooling:__init(nPlanes, kW, kH, dW, dH)

  parent.__init(self)
  self.kW = kW or 2
  self.kH = kH or 2
  self.dW = dW or 2
  self.dH = dH or 2
  self.lambda=torch.Tensor(nPlanes):fill(0);
  self.alpha=torch.Tensor(nPlanes):fill(0);
  self.gradLambda=torch.Tensor(nPlanes);
  self.gradAlpha=torch.Tensor(nPlanes);

  --self:reset(nPlanes)

  self:cuda()
end
function SpatialInverseBilateralPooling:reset(nPlanes)

   stdv = 1./math.sqrt(nPlanes)

   self.lambda:uniform(-stdv, stdv)
   self.alpha:uniform(-stdv, stdv)
   self.gradLambda:zero()
   self.gradAlpha:zero()
end

function SpatialInverseBilateralPooling:updateOutput(input)

 self.output:resizeAs(input[2]):copy(input[2])


 -- assert(torch.isTypeOf(input[1], 'torch.CudaTensor'))
  --assert(torch.isTypeOf(self.lambda, 'torch.CudaTensor'))
  --assert(torch.isTypeOf(self.alpha, 'torch.CudaTensor'))
  --assert(torch.isTypeOf(input[2], 'torch.CudaTensor'))
  
  C.SpatialInverseBilateralPooling_updateOutput(cutorch.getState(), input[1]:cdata(),input[2]:cdata(), self.lambda:cdata(), self.alpha:cdata(), self.output:cdata(),
  self.kW, self.kH, self.dW, self.dH)
  return self.output
end

function SpatialInverseBilateralPooling:updateGradInput(input, gradOutput)
self.gradInput={torch.CudaTensor(input[1]:size()):zero(),torch.CudaTensor(input[2]:size()):zero()}
  --assert(torch.isTypeOf(input[1], 'torch.CudaTensor'))
  --assert(torch.isTypeOf(self.lambda, 'torch.CudaTensor'))
  --assert(torch.isTypeOf(self.alpha, 'torch.CudaTensor'))
  --assert(torch.isTypeOf(input[2], 'torch.CudaTensor'))
  --assert(torch.isTypeOf(gradOutput, 'torch.CudaTensor'))
  
  C.SpatialInverseBilateralPooling_updateGradInput(cutorch.getState(), input[1]:cdata(),input[2]:cdata(),self.lambda:cdata(),self.alpha:cdata(), self.gradInput[1]:cdata(), self.gradInput[2]:cdata(),
  	gradOutput:cdata(), self.kW, self.kH, self.dW, self.dH)

--if torch.any(self.gradInput[1]:ne(self.gradInput[1])) then print('gradI has nan') end
--if torch.any(self.gradInput[2]:ne(self.gradInput[2])) then print('gradIt has nan') end 
 return self.gradInput
end

function SpatialInverseBilateralPooling:accGradParameters(input, gradOutput, scale)
 scale = scale or 1
  self.gradLambda:zero()
  self.gradAlpha:zero()
  --local gradLambda=torch.Tensor(lambda:size())
  C.SpatialInverseBilateralPooling_accGradParameters(cutorch.getState(), input[1]:cdata(),input[2]:cdata(),self.lambda:cdata(),self.alpha:cdata(), self.gradLambda:cdata(), self.gradAlpha:cdata(),
  	gradOutput:cdata(), self.kW, self.kH, self.dW, self.dH)

--if torch.any(self.gradLambda:ne(self.gradLambda)) then print('gradLambda has NaN') end
--if torch.any(self.gradAlpha:ne(self.gradAlpha)) then print('gradAlpha has NaN') end
end

function SpatialInverseBilateralPooling:parameters()
  --return {self.lambda}, {self.gradLambda}
  return {self.lambda, self.alpha}, {self.gradLambda, self.gradAlpha}
end
-------------------------------------------------------------------------------------------------------------
return visinf.SpatialInverseBilateralPooling
