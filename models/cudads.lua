--  Copyright (c) 2018, TU Darmstadt.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree.

require 'paths'
require 'nn'
require 'cunn'
require 'visinf'
function FilterBankShared(nPlanes)
   local p=nn.Parallel(2,2)
        local t1 =nn.Sequential();t1:add(nn.Unsqueeze(2));t1:add(nn.SpatialConvolution(1,1,3,3,2,2,1,1))
         p:add(t1)
   for i=2,nPlanes do



  
         --local t=nn.Sequential();t:add(nn.Unsqueeze(2));t:add(nn.SpatialConvolution(1,1,3,3,2,2,1,1):noBias():share(t1.modules[2],'weight'))
         p:add(t1:clone('weight','gradWeight'))
     
   end
   return p
end

function FilterBank(nPlanes)
   local p=nn.Parallel(2,2)
   for i=1,nPlanes do
   local t=nn.Sequential();t:add(nn.Unsqueeze(2));t:add(nn.SpatialConvolution(1,1,3,3,2,2,1,1):noBias())
      p:add(t)
   end
   return p
end

function FilterDownSample(nPlanes)

   naive=nn.Sequential():add(FilterBank(nPlanes)):add(nn.SpatialBatchNormalization(nPlanes,1e-5))
   a=nn.ConcatTable():add(nn.Identity()):add(naive)
   block=nn.Sequential():add(a):add(visinf.SpatialInverseBilateralPooling(nPlanes))
   return block:cuda()
end
function AverageDownSample(nPlanes)
   
   naive=nn.SpatialAveragePooling(3,3,2,2,1,1)
   a=nn.ConcatTable():add(nn.Identity()):add(naive)
   block=nn.Sequential():add(a):add(visinf.SpatialInverseBilateralPooling(nPlanes))
  
   return block:cuda()
end

