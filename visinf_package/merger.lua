require 'nn'
local merger, parent = torch.class('visinf.merger', 'nn.Module')


function merger:__init()
  parent.__init(self)
  self.size1=1
  self.size2=1
  self.size3=1
  self.size4=1
  self:cuda()
end


function merger:updateOutput(input)
  self.size1=input[1]:size(1)
  self.size2=input[1]:size(2)
  self.size3=input[1]:size(3)
  self.size4=input[1]:size(4)

local temp1=torch.cat(input[1]:view(self.size1,self.size2,self.size3,1,self.size4,1),input[3]:view(self.size1,self.size2,self.size3,1,self.size4,1),4)
  local temp2=torch.cat(input[2]:view(self.size1,self.size2,self.size3,1,self.size4,1),input[4]:view(self.size1,self.size2,self.size3,1,self.size4,1),4)
  self.output=torch.cat(temp1,temp2,6):view(self.size1,self.size2,2*self.size3,2*self.size4)
  return self.output
end

function merger:updateGradInput(input, gradOutput)
  self.gradInput={}


  self.gradInput[1]=gradOutput:view(self.size1,self.size2,self.size3,2,self.size4,2)[{{},{},{},1,{},1}]:contiguous():view(self.size1,self.size2,self.size3,self.size4):clone()
  self.gradInput[2]=gradOutput:view(self.size1,self.size2,self.size3,2,self.size4,2)[{{},{},{},1,{},2}]:contiguous():view(self.size1,self.size2,self.size3,self.size4):clone()
  self.gradInput[3]=gradOutput:view(self.size1,self.size2,self.size3,2,self.size4,2)[{{},{},{},2,{},1}]:contiguous():view(self.size1,self.size2,self.size3,self.size4):clone()
  self.gradInput[4]=gradOutput:view(self.size1,self.size2,self.size3,2,self.size4,2)[{{},{},{},2,{},2}]:contiguous():view(self.size1,self.size2,self.size3,self.size4):clone()

  return self.gradInput
end
