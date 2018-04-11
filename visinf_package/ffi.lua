local ffi = require 'ffi'

local libpath = package.searchpath('libvisinf', package.cpath)
if not libpath then return end

require 'cunn'

ffi.cdef[[
void SpatialStochasticPooling_updateOutput(THCState* state, THCudaTensor* input, 
    THCudaTensor* output, THCudaTensor* indices, int kW, int kH, int dW, int dH, bool train);
void SpatialStochasticPooling_updateGradInput(THCState* state, THCudaTensor* input,
    THCudaTensor* gradInput, THCudaTensor* gradOutput, THCudaTensor* indices, int kW, int kH, int dW, int dH);

void SpatialInverseBilateralPooling_updateOutput(THCState* state, THCudaTensor* I, THCudaTensor* It, THCudaTensor* lambda, THCudaTensor* alpha, 
    THCudaTensor* output, int kW, int kH, int dW, int dH);
void SpatialInverseBilateralPooling_updateGradInput(THCState* state, THCudaTensor* I, THCudaTensor* It, THCudaTensor* lambda, THCudaTensor* alpha,
    THCudaTensor* gradI, THCudaTensor* gradIt, THCudaTensor* gradOutput, int kW, int kH, int dW, int dH);
void SpatialInverseBilateralPooling_accGradParameters(THCState* state, THCudaTensor* I,THCudaTensor* It, THCudaTensor* lambda,THCudaTensor* alpha,
    THCudaTensor* lambda_gradient, THCudaTensor* alpha_gradient, THCudaTensor* gradOutput, int kW, int kH, int dW, int dH);
]]

return ffi.load(libpath)
