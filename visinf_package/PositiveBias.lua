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
local PositiveBias, Parent = torch.class('visinf.PositiveBias','nn.Module')


--------------------------------------------------------------------------------
function PositiveBias:__init(nPlane)
  Parent.__init(self)
  self.weight = torch.Tensor(nPlane)
  self.gradWeight = torch.Tensor(nPlane):zero()
  self:reset()

end
------------------------------------------------------------------------------------------
function PositiveBias:reset(stdv)
   if stdv then
      --std of uniform distribution on interval [-a,a] = a/sqrt(3)
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1.0/math.sqrt(self.weight:nElement())
   end
   self.weight:uniform(-stdv,stdv)
return self

end
-----------------------------------------------------------------------------------
--function:forward
function PositiveBias:updateOutput(input)
self.weight[self.weight:ge(10)]=10;
self.output:resizeAs(input):copy(input)

  self.output:add(torch.exp(self.weight):view(1,self.weight:nElement(),1,1):expandAs(self.output))

  return self.output
end
-------------------------------------------------------------------------------------------




-- function: backward


function PositiveBias:updateGradInput(input,gradOutput)


self.gradInput:resizeAs(gradOutput):copy(gradOutput)


  return self.gradInput
end
-------------------------------------------------------------------------------------------------------------------------------------------
 function PositiveBias:accGradParameters(input, gradOutput, scale)
 scale = scale or 1
--Charbonnier
self.gradWeight:add(torch.exp(self.weight):view(1,self.weight:nElement(),1,1):expandAs(self.output):cmul(gradOutput):sum(1):sum(3):sum(4):mul(scale))
  

end
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--function PositiveBias:parameters()

  --return {self.weight}, {self.gradWeight}
--end
-------------------------------------------------------------------------------------------------------------
return visinf.PositiveBias

