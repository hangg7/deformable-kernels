#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "nd_linear_sample_cuda.h"

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

#if !defined(_MSC_VER)
#define CUDA_UNROLL _Pragma("unroll")
#define CUDA_NOUNROLL _Pragma("nounroll")
#else
#define CUDA_UNROLL
#define CUDA_NOUNROLL
#endif

template <typename DType>
__device__ inline DType ldg(const DType* address) {
#if __CUDA_ARCH__ >= 350
    return __ldg(address);
#else
    return *address;
#endif
}

/*------------------------------------------------------------------------------------------------------------*/


// data: d1, d2, ..., dn
// shape: s1, s2, ..., sn
// coord: x1, x2, ..., xn
template <typename scalar_t>
inline __device__ scalar_t nd_linear(const scalar_t *data, const scalar_t* shape, const scalar_t* coord, const int dims) {
  for (int d = 0; d < dims; d++) {
    scalar_t x = ldg(coord + d);
    int s = static_cast<int>(ldg(shape + d));
    if (x <= -1 || x >= s) {
      return 0;
    }
  }

  const uint corners = 1 << dims;
  scalar_t val = 0;
  for (uint i = 0; i < corners; i++) {   
    int data_offset = 0;
    scalar_t data_weight = 1;   
    bool out_of_scope = false;

    for (uint d = 0; d < dims; d++) {
      scalar_t x = ldg(coord + d);
      int s = static_cast<int>(ldg(shape + d));

      data_offset *= s;
      uint offset = (i >> d) & 1;      
      int x_low = floor(x);
      scalar_t w = x - x_low;

      if (offset == 0 && x_low >= 0) {
        data_offset += x_low;
        data_weight *= 1-w;
      } 
      else if (offset == 1 && x_low + 1 <= s - 1) {
        data_offset += x_low + 1;
        data_weight *= w;
      } 
      else {
        out_of_scope = true;
        break;
      }
    }
    if (!out_of_scope)
      val += data_weight * ldg(data + data_offset);
  }
  return val;
}

template <typename scalar_t>
inline __device__ void nd_linear_backward_data(const scalar_t top_gradient, const scalar_t* shape, const scalar_t* coord, const int dims, scalar_t* data_gradient) {
  for (int d = 0; d < dims; d++) {
    scalar_t x = ldg(coord + d);
    int s = static_cast<int>(ldg(shape + d));
    if (x <= -1 || x >= s) {
      return;
    }
  }

  const uint corners = 1 << dims;
  for (uint i = 0; i < corners; i++) {   
    int data_offset = 0;
    scalar_t data_weight = 1;   
    bool out_of_scope = false;

    for (uint d = 0; d < dims; d++) {
      scalar_t x = ldg(coord + d);
      int s = static_cast<int>(ldg(shape + d));

      data_offset *= s;
      uint offset = (i >> d) & 1;      
      int x_low = floor(x);
      scalar_t w = x - x_low;

      if (offset == 0 && x_low >= 0) {
        data_offset += x_low;
        data_weight *= 1-w;
      } 
      else if (offset == 1 && x_low + 1 <= s - 1) {
        data_offset += x_low + 1;
        data_weight *= w;
      } 
      else {
        out_of_scope = true;
        break;
      }
    }
    if (!out_of_scope)
      atomicAdd(data_gradient + data_offset, data_weight * top_gradient);
  }
}

template <typename scalar_t>
inline __device__ scalar_t nd_linear_backward_coord(const scalar_t top_gradient, const scalar_t *data, const scalar_t* shape, const scalar_t* coord, const int dims, const int gdim) {
  for (int d = 0; d < dims; d++) {
    scalar_t x = ldg(coord + d);
    int s = static_cast<int>(ldg(shape + d));
    if (x <= -1 || x >= s) {
      return 0;
    }
  }

  scalar_t grad = 0;
  const uint corners = 1 << dims;
  for (uint i = 0; i < corners; i++) {   
    int data_offset = 0;
    scalar_t data_weight = 1;   
    bool out_of_scope = false;

    for (uint d = 0; d < dims; d++) {
      scalar_t x = ldg(coord + d);
      int s = static_cast<int>(ldg(shape + d));

      data_offset *= s;
      uint offset = (i >> d) & 1;      
      int x_low = floor(x);
      scalar_t w = x - x_low;

      if (offset == 0 && x_low >= 0) {
        data_offset += x_low;
        data_weight *= (d == gdim) ? scalar_t(-1) : 1-w;
      } 
      else if (offset == 1 && x_low + 1 <= s - 1) {
        data_offset += x_low + 1;
        data_weight *= (d == gdim) ? scalar_t(1) : w;
      } 
      else {
        out_of_scope = true;
        break;
      }
    }
    if (!out_of_scope) {
      grad += top_gradient * ldg(data + data_offset) *  data_weight;
    }
  }
  return grad;
}


/*------------------------------------------------------------------------------------------------------------*/


template<typename scalar_t>
__global__ void NdLinearSampleForwardKernel(int n,
                             const scalar_t* data,
                             const scalar_t *shape,
                             const scalar_t* coord,
                             const SampleArgs args,
                             scalar_t* output) {
   
  //const int batch = args.batch;                              
  const int channel = args.channel;
  const int spatial_dims = args.spatial_dims;
  const int prod_shape = args.prod_shape;
  
  CUDA_KERNEL_LOOP(thread_id, n) {
    const int out_c = thread_id % channel;
    const int out_n = thread_id / channel;
    output[thread_id] = nd_linear(data + out_c * prod_shape, shape, coord + out_n * spatial_dims, spatial_dims);
  }
}


void NdLinearSampleForward(const at::Tensor data, const at::Tensor shape, const at::Tensor coord, const SampleArgs args, at::Tensor output) {
  int num_kernels = args.batch * args.channel;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data.type(), "NdLinearSampleForward_GPU", ([&] {
        const scalar_t *data_ = data.data<scalar_t>();
        const scalar_t *shape_ = shape.data<scalar_t>();
        const scalar_t *coord_ = coord.data<scalar_t>();
        scalar_t *output_ = output.data<scalar_t>();

        NdLinearSampleForwardKernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_, shape_, coord_, args, output_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in NdLinearSampleForwardKernel: %s\n", cudaGetErrorString(err));
  }
}


template<typename scalar_t>
__global__ void NdLinearSampleBackwardDataKernel(int n,
                             const scalar_t* out_grad,
                             const scalar_t *shape,
                             const scalar_t* coord,
                             const SampleArgs args,
                             scalar_t* in_grad) {
   
  //const int batch = args.batch;                              
  const int channel = args.channel;
  const int spatial_dims = args.spatial_dims;
  const int prod_shape = args.prod_shape;
  
  CUDA_KERNEL_LOOP(thread_id, n) {
    const int out_c = thread_id % channel;
    const int out_n = thread_id / channel;
    nd_linear_backward_data(ldg(out_grad + thread_id), shape, coord + out_n * spatial_dims, spatial_dims, in_grad + out_c * prod_shape);
  }
}


void NdLinearSampleBackwardData(const at::Tensor out_grad, const at::Tensor shape, const at::Tensor coord, const SampleArgs args, at::Tensor in_grad) {
  int num_kernels = args.batch * args.channel;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "NdLinearSampleBackwardData_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *shape_ = shape.data<scalar_t>();
        const scalar_t *coord_ = coord.data<scalar_t>();
        scalar_t *in_grad_ = in_grad.data<scalar_t>();

        NdLinearSampleBackwardDataKernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, out_grad_, shape_, coord_, args, in_grad_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in NdLinearSampleBackwardDataKernel: %s\n", cudaGetErrorString(err));
  }
}


template<typename scalar_t>
__global__ void NdLinearSampleBackwardCoordKernel(int n,
                             const scalar_t* out_grad,
                             const scalar_t* data,
                             const scalar_t *shape,
                             const scalar_t* coord,
                             const SampleArgs args,
                             scalar_t* coord_grad_c) {
   
  //const int batch = args.batch;                              
  const int channel = args.channel;
  const int spatial_dims = args.spatial_dims;
  const int prod_shape = args.prod_shape;
  
  CUDA_KERNEL_LOOP(thread_id, n) {
    const int out_c = thread_id % channel;
    const int out_d = (thread_id / channel) % spatial_dims;
    const int out_n = (thread_id / channel) / spatial_dims;
    coord_grad_c[thread_id] = nd_linear_backward_coord(ldg(out_grad + out_n * channel + out_c), data + out_c * prod_shape, shape, coord + out_n * spatial_dims, spatial_dims, out_d);
  }
}


void NdLinearSampleBackwardCoord(const at::Tensor out_grad, const at::Tensor data, const at::Tensor shape, const at::Tensor coord, const SampleArgs args, at::Tensor coord_grad_c) {
  int num_kernels = args.batch * args.spatial_dims * args.channel;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data.type(), "NdLinearSampleBackwardCoord_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *data_ = data.data<scalar_t>();
        const scalar_t *shape_ = shape.data<scalar_t>();
        const scalar_t *coord_ = coord.data<scalar_t>();
        scalar_t *coord_grad_c_ = coord_grad_c.data<scalar_t>();

        NdLinearSampleBackwardCoordKernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, out_grad_, data_, shape_, coord_, args, coord_grad_c_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in NdLinearSampleBackwardCoordKernel: %s\n", cudaGetErrorString(err));
  }
}
