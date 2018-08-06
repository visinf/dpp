#  Copyright (c) 2018, TU Darmstadt.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree.

# import tc and torch both
import tensor_comprehensions as tc
import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
# define the operation as TC language
lang = """
def PositivePowBias(float(B, C, W, H) I0, float(C) Alpha, float(C) Lambda) -> (O) {
    O(b, c, w, h) = pow(I0(b, c, w, h),exp(Lambda(c)))
    O(b, c, w, h) =  O(b, c, w, h) + exp(Alpha(c))
    }
def PositivePowBias_grad(float(B, C, W, H) I0, float(C) Alpha, float(C) Lambda, float(B,C,W,H) d_O) -> (d_I0, d_Lambda, d_Alpha){
d_I0(b,c,w,h) =  pow(I0(b,c,w,h), exp(Lambda(c))-1) * exp(Lambda(c))
d_Lambda(c) +=! pow(I0(b,c,w,h),exp(Lambda(c))) * log(I0(b,c,w,h))*exp(Lambda(c))
d_Alpha (c) = B*W*H*exp(Alpha(c))

}
"""
# register the lang with TC backend
PositivePowBias = tc.define(lang, training=True, name="PositivePowBias", backward="PositivePowBias_grad")
# create input cuda tensors
B,C,W,H = 32, 512, 32, 32
I0, Alpha, Lambda = torch.randn(B, C, W, H).cuda(), torch.randn(C).cuda(),torch.randn(C).cuda()
# choose the options that resemble the operation and run
#out = tensordot(I0, I1, options=tc.Options("conv"))
# autotune the kernel
best_options = PositivePowBias.autotune(I0, Alpha, Lambda, cache="PositivePowBias_32_512_32_32.tc",generations=2)
# run the kernel with the autotuned options
#out = PositivePowBias(I0, Alpha, Lambda, options=best_options)

class pospowbias(nn.Module):
    def __init__(self):
        super(pospowbias, self).__init__()
        self.Lambda = nn.Parameter(torch.zeros(1))
        self.Alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        return PositivePowBias(x, self.Alpha, self.Lambda, options=best_options)

class DPP(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pospowbias=pospowbias()
    def forward(self, I):
        It   = F.upsample(F.avg_pool2d(I, 2), scale_factor=2, mode='nearest')
        x   = ((I-It)**2)+1e-3
        xn = F.upsample(F.avg_pool2d(x, 2), scale_factor=2, mode='nearest')
        w  = pospowbias(x/xn)
        kp = F.avg_pool2d(w, 2)
        Iw = F.avg_pool2d(I*w, 2)
        return Iw/kp

