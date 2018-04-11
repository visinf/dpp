--  Copyright (c) 2018, TU Darmstadt.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree.


require 'paths'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nnlr'
require 'visinf.merger'
require 'visinf.s3pool'
require 'visinf.DPP_sym_lite'

local S3_DPP_sym_lite, Parent = torch.class('visinf.S3_DPP_sym_lite','nn.Sequential')


function S3_DPP_sym_lite:__init(nPlane)
   Parent.__init(self)

   local model=nn.Sequential()
   local block=nn.ConcatTable()
   local DPP1=visinf.DPP_sym_lite(nPlane)
   local DPP2=visinf.DPP_sym_lite(nPlane)
   local DPP3=visinf.DPP_sym_lite(nPlane)
   local DPP4=visinf.DPP_sym_lite(nPlane)

   DPP2.modules[1].modules[10]:share(DPP1.modules[1].modules[10],'weight','gradWeight')
   DPP3.modules[1].modules[10]:share(DPP1.modules[1].modules[10],'weight','gradWeight')
   DPP4.modules[1].modules[10]:share(DPP1.modules[1].modules[10],'weight','gradWeight')

   DPP2.modules[1].modules[11]:share(DPP1.modules[1].modules[11],'weight','gradWeight')
   DPP3.modules[1].modules[11]:share(DPP1.modules[1].modules[11],'weight','gradWeight')
   DPP4.modules[1].modules[11]:share(DPP1.modules[1].modules[11],'weight','gradWeight')



   local s1=nn.Sequential():add(DPP1)
   local s2=nn.Sequential():add(nn.SpatialReplicationPadding(-1,1,0,0)):add(DPP2)
   local s3=nn.Sequential():add(nn.SpatialReplicationPadding(0,0,-1,1)):add(DPP3)
   local s4=nn.Sequential():add(nn.SpatialReplicationPadding(-1,1,-1,1)):add(DPP4)
   block:add(s1):add(s2):add(s3):add(s4)
   merger=visinf.merger()
   model:add(block):add(merger):add(visinf.s3pool(2,2,2,2))
   self:add(model)
end




