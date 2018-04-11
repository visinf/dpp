--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

-- Code modified for DPP by Faraz Saeedan.
--
local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(paths.concat(opt.resume, 'model_latest.t7'))
   local optimState = torch.load(paths.concat(opt.resume, 'optim_latest.t7'))

   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, isBestModel, opt,bestTop1, bestTop5)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end
print('one GPU')
   -- create a clean copy on the CPU without modifying the original network
--   model = deepCopy(model):float():clearState()
   model = model:clone():float():clearState()
print('deep copy')
--   local modelFile = 'model_' .. epoch .. '.t7'
  -- local optimFile = 'optimState_' .. epoch .. '.t7'

   torch.save(paths.concat(opt.save, 'latest_model.t7'), model)
   print('model saved!')
   torch.save(paths.concat(opt.save, 'latest_optim.t7'), optimState)
   print('optimState saved!')
   torch.save(paths.concat(opt.save, 'latest.t7'), {
      epoch = epoch,
      bestTop1 = bestTop1,
      bestTop5 = bestTop5     
-- modelFile = modelFile,
     -- optimFile = optimFile,
   })
   print('epoch saved!')
   torch.save(paths.concat(opt.save, 'opt.t7'), opt)
   if isBestModel then
      torch.save(paths.concat(opt.save, 'model_best.t7'), model)
      print('best model saved!')
   end
end

return checkpoint
