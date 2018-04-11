
--  Copyright (c) 2018, TU Darmstadt.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree.

require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'nngraph'
local CWeightCalc, Parent = torch.class('visinf.CWeightCalc','nn.Module')


--------------------------------------------------------------------------------
function CWeightCalc:__init(nPlane)
  Parent.__init(self)
  self.weight = torch.Tensor(nPlane)
  self.gradWeight = torch.Tensor(nPlane)
  self:reset()
end
-----------------------------------------------------------------------------------------
function CWeightCalc:reset(stdv)
   if stdv then
      --std of uniform distribution on interval [-a,a] = a/sqrt(3)
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1.0/math.sqrt(self.weight:nElement())
   end
   self.weight:uniform(-stdv,stdv)

end
-----------------------------------------------------------------------------------
--function:forward
function CWeightCalc:updateOutput(input)
--self.weight[self.weight:ge(4)]=4;
self.output:resizeAs(input):copy(input)
self._lambda=torch.exp(self.weight):view(1,self.weight:nElement(),1,1):expandAs(input):div(2)
self.output:cpow(self._lambda)

  return self.output
end
-------------------------------------------------------------------------------------------




-- function: backward


function CWeightCalc:updateGradInput(input,gradOutput)


self.gradInput=torch.cdiv(self.output,input):cmul(self._lambda):cmul(gradOutput)


  return self.gradInput
end
-------------------------------------------------------------------------------------------------------------------------------------------
 function CWeightCalc:accGradParameters(input, gradOutput, scale)
scale = scale or 1
self.gradWeight:add(torch.cmul(self.output,torch.log(input)):cmul(gradOutput):cmul(self._lambda):sum(1):sum(3):sum(4):squeeze():mul(scale))


end
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--function CWeightCalc:parameters()

  --return {self.lambda}, {self.gradLambda}
--end
-------------------------------------------------------------------------------------------------------------
return visinf.CWeightCalc

