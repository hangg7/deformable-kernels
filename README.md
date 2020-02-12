# <b>Deformable Kernels</b> [[ICLR 2020]](https://arxiv.org/abs/1910.02940) [[Website]](https://people.eecs.berkeley.edu/~hangg/deformable-kernels/)

<img src='http://people.eecs.berkeley.edu/~hangg/projects/deformable-kernels/materials/erf_visualization.png' width=1024>

**Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation** <br>
[Hang Gao<sup>*</sup>](http://people.eecs.berkeley.edu/~hangg/), [Xizhou
Zhu<sup>*</sup>](https://scholar.google.com/citations?user=02RXI00AAAAJ&hl=en&oi=ao), [Steve Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en&oi=ao), [Jifeng Dai](https://jifengdai.org/). <br>
In [ICLR, 2020](https://arxiv.org/abs/1910.02940).

This repository contains official implementation of deformable kernels. <br>

**Table of contents**<br>
1. [Customized operators for deformable kernels, along with its variants.](#1-customized-operators)<br>
2. [Instructions to use our operators.](#2-quickstart)<br>
3. [Results on ImageNet & COCO benchmarks, with pretrained models for
reproduction.](#3-results--pretrained-models)<br>
5. [Training and evaluation code.](#4-training--evaluation-code)<br>

## (0) Getting started

### PyTorch
- Get [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
  installed on your machine.
- Install PyTorch ([pytorch.org](http://pytorch.org)).
- `conda env create -f environment.yml`.

### Apex
- Install [Apex](https://github.com/NVIDIA/apex/) from its official repo. This
  will require CUDA 10.1 to work with the latest pytorch version (which is
`pytorch=1.3.1` as being tested against). It is used for fast mix-precision
inference and should work out of the box.

### Compile our operators
```bash
# assume at project root
(
cd deformable_kernels/ops/deform_kernel;
pip install -e .;
)
```


## (1) Customized operators

<img src='http://people.eecs.berkeley.edu/~hangg/projects/deformable-kernels/materials/dk_forward.png' width=768>

This repo includes all deformable kernel variants described in our paper, namely:

- Global Deformable Kernels;
- Local Deformable Kernels;
- Local Deformable Kernels integrating with Deformable Convolutions;

Instead of learning offsets on image space, we propose to deform and resample
on kernel space. This enables powerful dynamic inference capacity. For more
technical details, please refer to their
[definitions](deformable_kernels/modules/deform_kernel.py).

We also provide implementations on our rivalries, namely:

- [Deformable Convolutions](https://arxiv.org/abs/1703.06211);
- [Soft Conditional Computation](https://arxiv.org/abs/1904.04971);

Please refer to their module definitions under `deformable_kernels/modules` folder.


## (2) Quickstart
The following snippet constructs the deformable kernels we used for our experiments

```python
from deformable_kernels.modules import (
    GlobalDeformKernel2d,
    DeformKernel2d,
    DeformKernelConv2d,
)

# global DK with scope size 2, kernel size 1, stride 1, padding 0, depthwise convolution.
gdk = GlobalDeformKernel2d((2, 2), [inplanes], [inplanes], groups=[inplanes])
# (local) DK with scope size 4, kernel size 3, stride 1, padding 1, depthwise convolution.
dk = DeformKernel2d((4, 4), [inplanes], [inplanes], 3, 1, 1, groups=[inplanes])
# (local) DK integrating with dcn, with kernel & image offsets separately learnt.
dkc = DeformKernelConv2d((4, 4), [inplanes], [inplanes], 3, 1, 1, groups=[inplanes]).
```

Note that all of our customized operators only support depthwise convolutions
now, mainly because that efficiently resampling kernels at runtime is
extremely slow if we orthogonally compute over each channel. We are trying to
loose this requirement by iterating our CUDA implementation. Any contribuitions
are welcome!


## (3) Results & pretrained models
Under construction.


## (4) Training & evaluation code
Under construction.


## (A) License
This project is released under the [MIT license](LICENSE).


## (B) Citation & Contact
If you find this repo useful for your research, please consider citing this
bibtex:

```tex
@article{gao2019deformable,
  title={Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation},
  author={Gao, Hang and Zhu, Xizhou and Lin, Steve and Dai, Jifeng},
  journal={arXiv preprint arXiv:1910.02940},
  year={2019}
}
```

Please contact Hang Gao `<hangg AT eecs DOT berkeley DOT com>` and Xizhou Zhu
`<ezra0408 AT mail.ustc DOT edu DOT cn>` with any comments or feedback.
