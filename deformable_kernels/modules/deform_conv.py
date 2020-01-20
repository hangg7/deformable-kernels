#!/usr/bin/env python3
#
# File   : deform_conv.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 01/17/2020
#
# Distributed under terms of the MIT license.

from .deform_kernel import DeformKernelConv2d

__all__ = ['DeformConv2d']


class DeformConv2d(DeformKernelConv2d):
    """
    Depthwise deformable convolution.
    """
    def __init__(
        self,
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
        super().__init__(
            (kernel_size, kernel_size),
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            offset_clip,
        )
