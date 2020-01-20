#!/usr/bin/env python3
#
# File   : filter_sample_depthwise.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 01/17/2020
#
# Distributed under terms of the MIT license.

import torch
from torch.autograd import Function
from apex import amp

from .. import filter_sample_depthwise_cuda

__all__ = ['sample_depthwise', 'deform_sample_depthwise']


class SampleDepthwiseFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        rotation,
        weight,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        rotation_groups=1,
    ):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.rotation_groups = rotation_groups

        ctx.save_for_backward(input, rotation, weight)
        output = input.new_empty(
            SampleDepthwiseFunction._output_size(
                input, kernel_size, stride, padding, dilation))

        if not input.is_cuda:
            raise NotImplementedError
        else:
            filter_sample_depthwise_cuda.sample_depthwise_forward_cuda(
                input, rotation, weight, output,
                ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1],
                ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                weight.size(2), weight.size(3), rotation_groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, rotation, weight = ctx.saved_tensors

        grad_input = grad_rotation = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:

            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                filter_sample_depthwise_cuda.sample_depthwise_backward_data_cuda(
                    grad_output, input, rotation, weight, grad_input,
                    ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1],
                    ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                    weight.size(2), weight.size(3), ctx.rotation_groups)

            if ctx.needs_input_grad[1]:
                grad_rotation = torch.zeros_like(rotation)
                filter_sample_depthwise_cuda.sample_depthwise_backward_rotation_cuda(
                    grad_output, input, rotation, weight, grad_rotation,
                    ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1],
                    ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                    weight.size(2), weight.size(3), ctx.rotation_groups)

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                filter_sample_depthwise_cuda.sample_depthwise_backward_filter_cuda(
                    grad_output, input, rotation, weight, grad_weight,
                    ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1],
                    ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                    weight.size(2), weight.size(3), ctx.rotation_groups)

        return (grad_input, grad_rotation, grad_weight,) + (None,) * 4

    @staticmethod
    def _output_size(input, kernel_size, stride, padding, dilation):
        output_size = (input.size(0), input.size(1))
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (kernel_size[d] - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


class DeformableSampleDepthwiseFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        rotation,
        weight,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        rotation_groups=1,
    ):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.rotation_groups = rotation_groups

        ctx.save_for_backward(input, offset, rotation, weight)
        output = input.new_empty(
            DeformableSampleDepthwiseFunction._output_size(
                input, kernel_size, stride, padding, dilation))

        if not input.is_cuda:
            raise NotImplementedError
        else:
            filter_sample_depthwise_cuda. \
                deformable_sample_depthwise_forward_cuda(
                    input, offset, rotation, weight, output,
                    ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0],
                    ctx.stride[1], ctx.padding[0], ctx.padding[1],
                    ctx.dilation[0], ctx.dilation[1], weight.size(2),
                    weight.size(3), rotation_groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, rotation, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_rotation = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                filter_sample_depthwise_cuda. \
                    deformable_sample_depthwise_backward_data_cuda(
                        grad_output, input, offset, rotation, weight,
                        grad_input, ctx.kernel_size[0], ctx.kernel_size[1],
                        ctx.stride[0], ctx.stride[1], ctx.padding[0],
                        ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                        weight.size(2), weight.size(3), ctx.rotation_groups)

            if ctx.needs_input_grad[1]:
                grad_offset = torch.zeros_like(offset)
                filter_sample_depthwise_cuda. \
                    deformable_sample_depthwise_backward_offset_cuda(
                        grad_output, input, offset, rotation, weight,
                        grad_offset, ctx.kernel_size[0], ctx.kernel_size[1],
                        ctx.stride[0], ctx.stride[1], ctx.padding[0],
                        ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                        weight.size(2), weight.size(3), ctx.rotation_groups)

            if ctx.needs_input_grad[2]:
                grad_rotation = torch.zeros_like(rotation)
                filter_sample_depthwise_cuda. \
                    deformable_sample_depthwise_backward_rotation_cuda(
                        grad_output, input, offset, rotation, weight,
                        grad_rotation, ctx.kernel_size[0], ctx.kernel_size[1],
                        ctx.stride[0], ctx.stride[1], ctx.padding[0],
                        ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                        weight.size(2), weight.size(3), ctx.rotation_groups)

            if ctx.needs_input_grad[3]:
                grad_weight = torch.zeros_like(weight)
                filter_sample_depthwise_cuda. \
                    deformable_sample_depthwise_backward_filter_cuda(
                        grad_output, input, offset, rotation, weight,
                        grad_weight, ctx.kernel_size[0], ctx.kernel_size[1],
                        ctx.stride[0], ctx.stride[1], ctx.padding[0],
                        ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
                        weight.size(2), weight.size(3), ctx.rotation_groups)

        return (grad_input, grad_offset, grad_rotation, grad_weight,) + (None,) * 4

    @staticmethod
    def _output_size(input, kernel_size, stride, padding, dilation):
        output_size = (input.size(0), input.size(1))
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (kernel_size[d] - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


# register as fp32 functions.
sample_depthwise = amp.float_function(SampleDepthwiseFunction.apply)
deform_sample_depthwise = amp.float_function(DeformableSampleDepthwiseFunction.apply)
