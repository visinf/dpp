require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
local s3pool, parent = torch.class('visinf.s3pool', 'nn.Module')


function s3pool:__init(kW, kH, dW, dH)
  parent.__init(self)
  self.train = true
  self.dH=dH;
  self.dW=dW;
  self.avgpool=cudnn.SpatialAveragePooling(2,2,2,2);
  self.stage1=torch.Tensor(1);
  self.indicesH=torch.LongTensor(1);
  self.indicesW=torch.LongTensor(1);
  self.grid=2;
  self.indices=torch.Tensor(1);
  self:cuda()
end


function s3pool:updateOutput(input)
     self.grid=input:size(3) / 2;
  if self.train then 
     local gridH   = input:size(3) / self.grid;
     local gridW   = input:size(4) / self.grid;
     self.indicesH:resize(input:size(3) / self.dH)
     self.indicesW:resize(input:size(4) / self.dW)
     for i=1,gridH do
      
        self.indicesH[{{((i-1)*(self.grid/self.dH))+1,(i*(self.grid/self.dH))}}]=torch.multinomial(torch.ones(self.grid),self.grid/self.dH)+((i-1)*self.grid);
     end 
     self.indicesH=torch.sort(self.indicesH);
     for i=1,gridH do
      
        self.indicesW[{{((i-1)*(self.grid/self.dW))+1,(i*(self.grid/self.dW))}}]=torch.multinomial(torch.ones(self.grid),self.grid/self.dW)+((i-1)*self.grid);
     end 
     self.indicesW=torch.sort(self.indicesW);
     self.output=input:index(3,self.indicesH):index(4,self.indicesW);
  else
     self.output=self.avgpool:forward(input);
  end
  return self.output
end

function s3pool:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  for i=1,self.indicesH:nElement() do
     for j=1,self.indicesW:nElement() do
        self.gradInput[{{},{},self.indicesH[{i}],self.indicesW[{j}]}]=gradOutput[{{},{},i,j}]
     end
  end
  return self.gradInput
end
