#!/usr/bin/env python3
#
# File   : nd_linear_sample.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 01/17/2020
#
# Distributed under terms of the MIT license.

import torch
from torch.autograd import Function

from .. import nd_linear_sample_cuda

__all__ = ['nd_linear_sample']


class NdLinearSampleFunction(Function):
    @staticmethod
    def forward(ctx, input, coord):
        ctx.save_for_backward(input, coord)
        shape = torch.tensor(input.shape[1:], dtype=input.dtype, device=input.device)
        output = input.new_empty(NdLinearSampleFunction._output_size(input, coord))

        if not input.is_cuda:
            raise NotImplementedError
        else:
            nd_linear_sample_cuda.nd_linear_sample_forward_cuda(
                input, shape, coord, output
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, coord = ctx.saved_tensors
        shape = torch.tensor(input.shape[1:], dtype=input.dtype, device=input.device)
        grad_input = grad_coord = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if ctx.needs_input_grad[0]:
                grad_input = input.new_empty(*input.size())
                nd_linear_sample_cuda.nd_linear_sample_backward_data_cuda(
                    grad_output, input, shape, coord, grad_input
                )

            if ctx.needs_input_grad[1]:
                grad_coord_c = coord.new_empty(
                    coord.size(0), coord.size(1), input.size(0)
                )
                nd_linear_sample_cuda.nd_linear_sample_backward_coord_cuda(
                    grad_output, input, shape, coord, grad_coord_c
                )
                grad_coord = grad_coord_c.sum(2)

        return (grad_input, grad_coord)

    @staticmethod
    def _output_size(input, coord):
        output_size = (coord.size(0), input.size(0))
        return output_size


nd_linear_sample = NdLinearSampleFunction.apply
