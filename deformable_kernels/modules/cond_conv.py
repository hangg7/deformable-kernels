#!/usr/bin/env python3
#
# File   : cond_conv.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 12/25/2019
#
# Distributed under terms of the MIT license.

import torch

from apex import amp
from torch import nn


class CondConv2d(nn.Module):

    def __init__(self, num_experts, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        assert not bias

        self.weight = nn.Parameter(
            torch.tensor(
                num_experts * out_channels,
                in_channels // self.groups,
                kernel_size,
                kernel_size,
            )
        )
        self.fc = nn.Linear(in_channels, num_experts)
        self.fc.zero_init = True

    @amp.float_function
    def dynaic_inference(self, x, weight):
        # TODO(Hang Gao @ 12/26): make sure passing weight to amp is necessary.
        n = x.shape[0]

        avg_x = x.mean((2, 3))
        gate_x = torch.sigmoid(self.fc(avg_x))

        weight = torch.mm(
            gate_x,
            self.weight.reshape(self.num_experts, -1)
        ).reshape(
            n * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        return weight

    def forward(self, x):
        n, _, h, w = x.shape
        weight = self.dynaic_inference(x, self.weight)

        out = nn.functional.conv2d(
            x.reshape(1, n * self.in_channels, h, w),
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n*self.groups,
            padding_mode=self.padding_mode,
        )
        out = out.reshape(n, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{num_experts}, {in_channels}, {out_channels}'
             ', kernel_size={kernel_size}, stride={stride}'
             ', scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
