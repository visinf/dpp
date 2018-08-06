#  Copyright (c) 2018, TU Darmstadt.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree.


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensor_comprehensions as tc
from torchvision import datasets, transforms


lang = """
def PositivePowBias(float(B, C, W, H) I0, float(C) Alpha, float(C) Lambda) -> (O) {
    O(b, c, w, h) = pow(I0(b, c, w, h),exp(Lambda(c)))
    O(b, c, w, h) =  O(b, c, w, h) + exp(Alpha(c))
    }
"""
# register the lang with TC backend
PositivePowBias = tc.define(lang, name="PositivePowBias")
out1 = PositivePowBias(mat1, mat2, cache="PositivePowBias_32_512_32_32.tc")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg1 = nn.AvgPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, I):
        It= F.upsample(F.avg_pool2d(I, 2), scale_factor=2, mode='nearest')
        x= ((I-It)**2)+1e-3
        xn= F.upsample(F.avg_pool2d(x, 2), scale_factor=2, mode='nearest')
        w=x/xn
        return F.log_softmax(x, dim=1)
