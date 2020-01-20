#!/usr/bin/env python3
#
# File   : deform_kernel.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 01/17/2020
#
# Distributed under terms of the MIT license.

import torch
from torch import nn
from apex import amp

from ..ops.deform_kernel.functions import nd_linear_sample
from ..ops.deform_kernel.modules import (
    SampleDepthwise,
    DeformableSampleDepthwise,
)

__all__ = [
    'GlobalDeformKernel2d',
    'LocalDeformKernel2d',
    'DeformKernel2d',
    'DeformKernelConv2d',
]


class GlobalDeformKernel2d(nn.Module):

    def __init__(
        self,
        weight_shape,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_shape = weight_shape
        self.weight_dilate = 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.group = groups
        assert not bias

        self.weight = nn.Parameter(
            torch.Tensor(out_planes, in_planes // self.group, *self.weight_shape)
        )
        self.fc = nn.Linear(
            in_planes, kernel_size * kernel_size * len(self.weight_shape)
        )
        self.fc.zero_init = True

        assert len(self.weight_shape) >= 2

        start_h = (weight_shape[0] - (kernel_size - 1) * self.weight_dilate - 1) / 2.0
        start_w = (weight_shape[1] - (kernel_size - 1) * self.weight_dilate - 1) / 2.0
        self.fc_bias = []
        for h in range(kernel_size):
            for w in range(kernel_size):
                self.fc_bias += [
                    start_h + h * self.weight_dilate,
                    start_w + w * self.weight_dilate,
                ]
                for i in range(len(self.weight_shape) - 2):
                    self.fc_bias += [(self.weight_shape[i + 2] - 1) / 2.0]

    @amp.float_function
    def dynamic_weight(self, x, weight):
        n, c, h, w = x.shape
        avg_x = x.view(n, c, -1).mean(2)
        coord = self.fc(avg_x) * self.weight_dilate + torch.tensor(
            self.fc_bias, dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        coord = torch.clamp(coord, 0, self.weight_shape[0] - 1)

        weight = weight.view(
            self.out_planes * self.in_planes // self.group, *self.weight_shape
        )
        coord = coord.view(
            n * self.kernel_size * self.kernel_size, len(self.weight_shape)
        )

        weight_sample = nd_linear_sample(weight, coord).view(
            n,
            self.kernel_size * self.kernel_size,
            self.out_planes * self.in_planes // self.group,
        )
        weight = weight_sample.transpose(1, 2).reshape(
            n * self.out_planes,
            self.in_planes // self.group,
            self.kernel_size,
            self.kernel_size,
        )
        return weight

    def forward(self, x):
        n, c, h, w = x.shape
        weight = self.dynamic_weight(x, self.weight)

        out = nn.functional.conv2d(
            x.view(1, n * c, h, w),
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n * self.group,
        )
        out = out.view(n, self.out_planes, out.shape[2], out.shape[3])
        return out

    def extra_repr(self):
        s = (
            "{in_planes}, {out_planes}, weight_shape={weight_shape}, "
            "kernel_size={kernel_size}, stride={stride}, "
            "weight_dilate={weight_dilate}"
        )
        if self.padding != 0:
            s += ", padding={padding}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.group != 1:
            s += ", group={group}"
        return s.format(**self.__dict__)


class LocalDeformKernel2d(nn.Module):

    def __init__(
        self,
        weight_shape,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        rotation_groups=1,
        bias=False,
        rotation_clip=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_shape = weight_shape
        self.weight_dilate = 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.rotation_groups = rotation_groups
        self.group = groups
        assert not bias
        assert len(self.weight_shape) >= 2

        self.rotation_conv = nn.Conv2d(
            in_planes, rotation_groups * kernel_size * kernel_size * 2,
            kernel_size, stride, padding, dilation, bias=True
        )
        self.rotation_conv.zero_init = True
        self.rotation_clip = rotation_clip

        self.inner_conv = SampleDepthwise(
            weight_shape,
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            rotation_groups=rotation_groups,
            bias=bias,
        )

    def _clip_rotation(self, rotation):
        if isinstance(self.rotation_clip, tuple):
            return rotation.clamp(**self.rotation_clip)
        elif self.rotation_clip == 'scope':
            if not hasattr(self, 'fc_bias'):
                start_h = (self.weight_shape[0] - (self.kernel_size - 1) *
                           self.weight_dilate - 1) / 2.0
                start_w = (self.weight_shape[1] - (self.kernel_size - 1) *
                           self.weight_dilate - 1) / 2.0
                fc_bias = []
                for h in range(self.kernel_size):
                    for w in range(self.kernel_size):
                        fc_bias += [
                            start_h + h * self.weight_dilate,
                            start_w + w * self.weight_dilate,
                        ]
                        for i in range(len(self.weight_shape) - 2):
                            fc_bias += [(self.weight_shape[i + 2] - 1) / 2]
                self.fc_bias = rotation.new_tensor(fc_bias) \
                    .repeat(self.rotation_groups)[None, :, None, None]
            coord = (rotation * self.weight_dilate + self.fc_bias).clamp(
                0, self.weight_shape[0] - 1)
            return (coord - self.fc_bias) / self.weight_dilate
        else:
            raise NotImplementedError(
                f'Expect rotation_clip to be tuple or "scope", '
                f'but get {self.rotation_clip}'
            )

    def forward(self, x):
        rotation = self.rotation_conv(x)
        if self.rotation_clip is not None:
            rotation = self._clip_rotation(rotation)
        rotation *= self.weight_dilate
        out = self.inner_conv(x, rotation)

        return out


# refer to local deformable kernel as the default.
DeformKernel2d = LocalDeformKernel2d


class DeformKernelConv2d(nn.Module):

    def __init__(
        self,
        weight_shape,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        offset_clip=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.group = groups
        assert not bias

        self.offset_conv = nn.Conv2d(
            in_planes, kernel_size * kernel_size * 2,
            kernel_size, stride, padding, dilation, bias=True
        )
        self.offset_conv.zero_init = True
        self.offset_clip = offset_clip

        self.inner_conv = DeformableSampleDepthwise(
            weight_shape,
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        if self.offset_clip is not None:
            offset = offset.clamp(**self.offset_clip)
        offset *= self.dilation

        rotation = None
        out = self.inner_conv(x, offset, rotation)

        return out
