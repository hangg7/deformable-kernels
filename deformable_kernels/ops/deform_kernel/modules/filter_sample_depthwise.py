#!/usr/bin/env python3
#
# File   : filter_sample_depthwise.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 01/17/2020
#
# Distributed under terms of the MIT license.

import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..functions.filter_sample_depthwise import (
    sample_depthwise,
    deform_sample_depthwise,
)

__all__ = [
    'SampleDepthwise',
    'DeformableSampleDepthwise',
]


class SampleDepthwise(nn.Module):
    def __init__(self,
                 scope_size,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 rotation_groups=1,
                 bias=False):
        super(SampleDepthwise, self).__init__()
        self.in_channels = in_channels
        assert in_channels == out_channels and groups == in_channels
        assert in_channels % rotation_groups == 0 and \
            out_channels % rotation_groups == 0
        self.scope_size = scope_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.rotation_groups = rotation_groups
        self.bias = bias

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, 1, *self.scope_size))
        if not self.bias:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.Tensor(self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = 1
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, rotation=None):
        if rotation is None:
            output_size = self._output_size(input, self.weight)
            rotation = input.new_zeros(
                input.size(0),
                self.rotation_groups * self.kernel_size[0] *
                self.kernel_size[1] * 2,
                output_size[2],
                output_size[3])
        out = sample_depthwise(
            input, rotation, self.weight, self.kernel_size, self.stride,
            self.padding, self.dilation, self.rotation_groups)
        if self.bias is not None:
            out += self.bias.view(1, self.in_channels, 1, 1)
        return out

    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (self.kernel_size[d] - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size

    def extra_repr(self):
        s = ('scope_size={scope_size}, in_channels={in_channels}, '
             'kernel_size={kernel_size}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.rotation_groups != 1:
            s += ', rotation_groups={rotation_groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class DeformableSampleDepthwise(nn.Module):
    def __init__(self,
                 scope_size,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 rotation_groups=1,
                 bias=False):
        super(DeformableSampleDepthwise, self).__init__()
        self.in_channels = in_channels
        assert in_channels == out_channels and groups == in_channels
        assert in_channels % rotation_groups == 0 and \
            out_channels % rotation_groups == 0
        self.scope_size = scope_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.rotation_groups = rotation_groups
        self.bias = bias

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, 1, *self.scope_size))
        if not self.bias:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.Tensor(self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = 1
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, offset=None, rotation=None):
        if offset is None:
            output_size = self._output_size(input, self.weight)
            offset = input.new_zeros(
                input.size(0),
                self.kernel_size[0] * self.kernel_size[1] * 2,
                output_size[2],
                output_size[3])
        if rotation is None:
            output_size = self._output_size(input, self.weight)
            rotation = input.new_zeros(
                input.size(0),
                self.rotation_groups * self.kernel_size[0] *
                self.kernel_size[1] * 2,
                output_size[2],
                output_size[3])

        out = deform_sample_depthwise(
            input, offset, rotation, self.weight, self.kernel_size,
            self.stride, self.padding, self.dilation, self.rotation_groups)
        if self.bias is not None:
            out += self.bias.view(1, self.in_channels, 1, 1)
        return out

    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (self.kernel_size[d] - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size

    def extra_repr(self):
        s = ('scope_size={scope_size}, in_channels={in_channels}, '
             'kernel_size={kernel_size}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.rotation_groups != 1:
            s += ', rotation_groups={rotation_groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
