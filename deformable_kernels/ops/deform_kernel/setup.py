#!/usr/bin/env python3
#
# File   : setup.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 01/17/2020
#
# Distributed under terms of the MIT license.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='filter_sample_depthwise',
    ext_modules=[
        CUDAExtension(
            'filter_sample_depthwise_cuda',
            [
                'csrc/filter_sample_depthwise_cuda.cpp',
                'csrc/filter_sample_depthwise_cuda_kernel.cu',
            ]
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name="nd_linear_sample",
    ext_modules=[
        CUDAExtension(
            "nd_linear_sample_cuda",
            [
                "csrc/nd_linear_sample_cuda.cpp",
                "csrc/nd_linear_sample_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
