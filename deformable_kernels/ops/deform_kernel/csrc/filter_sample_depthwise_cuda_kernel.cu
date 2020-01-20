#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <stdio.h>
#include <math.h>
#include <float.h>

#include "filter_sample_depthwise_cuda.h"

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

template <typename scalar_t>
__device__ inline scalar_t ldg(const scalar_t* address) {
#if __CUDA_ARCH__ >= 350
    return __ldg(address);
#else
    return *address;
#endif
}

template <typename scalar_t>
inline scalar_t __device__ CudaMax(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

template <typename scalar_t>
inline scalar_t __device__ CudaMin(scalar_t a, scalar_t b) {
    return a < b ? a : b;
}

// assuming h, w is remainder of division, thus h in [0, height), w in [0, width)
template <typename scalar_t>
__device__ scalar_t planar_bilinear(
    const scalar_t *data,
    const int height,
    const int width,
    const scalar_t h,
    const scalar_t w) {

  if (h > -1 && w > -1 && h < height && w < width) {
    int h_low = floor(h);
    int w_low = floor(w);

    int h_high = h_low + 1;
    int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    scalar_t val = 0;
    if (h_low >= 0 && w_low >= 0)
      val += hh * hw * ldg(data + h_low * width + w_low);
    if (h_low >=0 && w_high <= width - 1)
      val += hh * lw * ldg(data + h_low * width + w_high);
    if (h_high <= height - 1 && w_low >= 0)
      val += lh * hw * ldg(data + h_high * width + w_low);
    if (h_high <= height - 1 && w_high <= width - 1)
      val += lh * lw * ldg(data + h_high * width + w_high);
    return val;
  } else {
    return 0;
  }
}

template <typename scalar_t>
__device__ void planar_bilinear_backward_data(
    const scalar_t partial_sum,
    const int height,
    const int width,
    const scalar_t h,
    const scalar_t w,
    scalar_t* filter_gradient) {

  if (h > -1 && w > -1 && h < height && w < width) {
    int h_low = floor(h);
    int w_low = floor(w);

    int h_high = h_low + 1;
    int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    if (h_low >= 0 && w_low >= 0)
      atomicAdd(filter_gradient + h_low * width + w_low, hh * hw * partial_sum);
    if (h_low >=0 && w_high <= width - 1)
      atomicAdd(filter_gradient + h_low * width + w_high, hh * lw * partial_sum);
    if (h_high <= height - 1 && w_low >= 0)
      atomicAdd(filter_gradient + h_high * width + w_low, lh * hw * partial_sum);
    if (h_high <= height - 1 && w_high <= width - 1)
      atomicAdd(filter_gradient + h_high * width + w_high, lh * lw * partial_sum);
  }
}

template <typename scalar_t>
__device__ scalar_t planar_bilinear_backward_coord(
    const scalar_t partial_sum,
    const scalar_t* filter,
    const int height,
    const int width,
    const scalar_t h,
    const scalar_t w,
    const int bp_dir) {

  if (h > -1 && w > -1 && h < height && w < width) {
    int h_low = floor(h);
    int w_low = floor(w);

    int h_high = h_low + 1;
    int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    if (bp_dir == 0) {
      scalar_t gradient_h = 0;
      if (h_low >= 0 && w_low >= 0)
        gradient_h -= hw * partial_sum * ldg(filter + h_low * width + w_low);
      if (h_low >=0 && w_high <= width - 1)
        gradient_h -= lw * partial_sum * ldg(filter + h_low * width + w_high);
      if (h_high <= height - 1 && w_low >= 0)
        gradient_h += hw * partial_sum * ldg(filter + h_high * width + w_low);
      if (h_high <= height - 1 && w_high <= width - 1)
        gradient_h += lw * partial_sum * ldg(filter + h_high * width + w_high);
      return gradient_h;
    } else {
      scalar_t gradient_w = 0;
      if (h_low >= 0 && w_low >= 0)
        gradient_w -= hh * partial_sum * ldg(filter + h_low * width + w_low);
      if (h_low >=0 && w_high <= width - 1)
        gradient_w += hh * partial_sum * ldg(filter + h_low * width + w_high);
      if (h_high <= height - 1 && w_low >= 0)
        gradient_w -= lh * partial_sum * ldg(filter + h_high * width + w_low);
      if (h_high <= height - 1 && w_high <= width - 1)
        gradient_w += lh * partial_sum * ldg(filter + h_high * width + w_high);
      return gradient_w;
    }
  } else {
    return 0;
  }
}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(
    const scalar_t *bottom_data,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = ldg(bottom_data + h_low * width + w_low);
  scalar_t v2 = 0;
  if (h_low >=0 && w_high <= width - 1)
    v2 = ldg(bottom_data + h_low * width + w_high);
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = ldg(bottom_data + h_high * width + w_low);
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = ldg(bottom_data + h_high * width + w_high);

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ void deformable_im2col_bilinear_backward(
    const scalar_t partial_sum,
    const scalar_t h,
    const scalar_t w,
    const int height,
    const int width,
    scalar_t* data_gradient) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  if (h_low >= 0 && w_low >= 0)
    atomicAdd(data_gradient + h_low * width + w_low, hh * hw * partial_sum);
  if (h_low >=0 && w_high <= width - 1)
    atomicAdd(data_gradient + h_low * width + w_high, hh * lw * partial_sum);
  if (h_high <= height - 1 && w_low >= 0)
    atomicAdd(data_gradient + h_high * width + w_low, lh * hw * partial_sum);
  if (h_high <= height - 1 && w_high <= width - 1)
    atomicAdd(data_gradient + h_high * width + w_high, lh * lw * partial_sum);

  return;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int height,
    const int width,
    const scalar_t *im_data,
    const int bp_dir) {

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * width + argmax_w_high];
  }

  return weight;
}

template<typename scalar_t, int kFilterHeight, int kFilterWidth>
__global__ __launch_bounds__(1024, 2) void SampleDepthwiseConv2dForwardKernel(
    int n,
    const scalar_t* input,
    const scalar_t* rotation_ratio,
    const scalar_t* filter,
    const SampleDepthwiseArgs args,
    scalar_t* output) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % channel;
    const int out_b = thread_id / out_width / out_height / channel;
    const int in_c = out_c;

    const int input_offset_temp =
      (out_b * channel + in_c) * (in_height * in_width);

    const int group_id = in_c % sampling_group;
    const int rotation_offset_temp =
      (out_b * sampling_group + group_id) * (filter_height * filter_width * 2) *
      out_height * out_width + (out_h * out_width + out_w);
    const int filter_offset_temp = in_c * scope_height * scope_width;

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    const int input_h_start = out_h * stride_height - pad_height;
    const int input_w_start = out_w * stride_width - pad_width;
    const int input_h_end = input_h_start + (filter_height - 1) * dilation_height;
    const int input_w_end = input_w_start + (filter_width - 1) * dilation_width;

    scalar_t sum = 0;
    if (input_h_start >= 0 && input_w_start >= 0 &&
        input_h_end < in_height && input_w_end < in_width) {
      // Loop that doesn't need to check for boundary conditions.
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h * dilation_height;

        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w * dilation_width;
          const int input_offset = (input_offset_temp) + (in_h * in_width) + in_w;

          const int rotation_offset_fhw = rotation_offset_temp +
            (f_h * filter_width + f_w) * 2 * out_height * out_width;
          const scalar_t rotation_ratio_h =
            ldg(rotation_ratio + rotation_offset_fhw);
          const scalar_t rotation_ratio_w =
            ldg(rotation_ratio + rotation_offset_fhw + out_height * out_width);

          const scalar_t filter_h = f_h + rotation_ratio_h +
            (scope_height - filter_height) / 2.0;
          const scalar_t filter_w = f_w + rotation_ratio_w +
            (scope_width - filter_width) / 2.0;
          sum += ldg(input + input_offset) * planar_bilinear(
              filter + filter_offset_temp,
              scope_height,
              scope_width,
              filter_h,
              filter_w);
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h * dilation_height;

        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w * dilation_width;

          // NOTE(Hang Gao @ 07/25): how much runtime will it save?
          if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
            const int input_offset = input_offset_temp + (in_h * in_width) + in_w;

            const int rotation_offset_fhw = rotation_offset_temp +
              (f_h * filter_width + f_w) * 2 * out_height * out_width;
            const scalar_t rotation_ratio_h =
              ldg(rotation_ratio + rotation_offset_fhw);
            const scalar_t rotation_ratio_w =
              ldg(rotation_ratio + rotation_offset_fhw + out_height * out_width);

            const scalar_t filter_h = f_h + rotation_ratio_h +
              (scope_height - filter_height) / 2.0;
            const scalar_t filter_w = f_w + rotation_ratio_w +
              (scope_width - filter_width) / 2.0;
            sum += ldg(input + input_offset) * planar_bilinear(
                filter + filter_offset_temp,
                scope_height,
                scope_width,
                filter_h,
                filter_w);
          }
        }

      }
    }
    output[thread_id] = sum;
  }
}

void SampleDepthwiseConv2dForward(
    const at::Tensor input,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor output) {

  int num_kernels = args.batch * args.channel * args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "SampleDepthwiseConv2dForward_GPU", ([&] {
        const scalar_t *input_ = input.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        const scalar_t *filter_ = filter.data<scalar_t>();
        scalar_t *output_ = output.data<scalar_t>();


        if (args.filter_height == 3 && args.filter_width == 3) {
          SampleDepthwiseConv2dForwardKernel<scalar_t, 3, 3>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels,
                input_,
                rotation_ratio_,
                filter_,
                args,
                output_);
        } else  {
          SampleDepthwiseConv2dForwardKernel<scalar_t, -1, -1>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels,
              input_,
              rotation_ratio_,
              filter_,
              args,
              output_);
        }

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SampleDepthwiseConv2dForwardKernel: %s\n", cudaGetErrorString(err));
  }
}

template<typename scalar_t, int kFilterHeight, int kFilterWidth>
__global__ __launch_bounds__(1024, 2) void DeformableSampleDepthwiseConv2dForwardKernel(
    int n,
    const scalar_t* input,
    const scalar_t* offset,
    const scalar_t* rotation_ratio,
    const scalar_t* filter,
    const SampleDepthwiseArgs args,
    scalar_t* output) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % channel;
    const int out_b = thread_id / out_width / out_height / channel;
    const int in_c = out_c;

    const int input_offset_temp =
      (out_b * channel + in_c) * (in_height * in_width);
    const int deformation_offset_temp =
      out_b * (filter_height * filter_width * 2) * out_height * out_width +
      (out_h * out_width + out_w);
    const int group_id = in_c % sampling_group;
    const int rotation_offset_temp = (out_b * sampling_group + group_id) *
      (filter_height * filter_width * 2) * out_height * out_width +
      (out_h * out_width + out_w);
    const int filter_offset_temp = in_c * scope_height * scope_width;

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    const int input_h_start = out_h * stride_height - pad_height;
    const int input_w_start = out_w * stride_width - pad_width;

    scalar_t sum = 0;
    CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
      const int in_h = input_h_start + f_h * dilation_height;

      CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
        const int in_w = input_w_start + f_w * dilation_width;
        const int deformation_offset_fhw = deformation_offset_temp +
          (f_h * filter_width + f_w) * 2 * out_height * out_width;
        const int rotation_offset_fhw = rotation_offset_temp +
          (f_h * filter_width + f_w) * 2 * out_height * out_width;

        const scalar_t input_h = in_h +
          ldg(offset + deformation_offset_fhw);
        const scalar_t input_w = in_w +
          ldg(offset + deformation_offset_fhw + out_height * out_width);

        if (input_h > -1 && input_w > -1 && input_h < in_height && input_w < in_width) {
          const scalar_t rotation_ratio_h =
            ldg(rotation_ratio + rotation_offset_fhw);
          const scalar_t rotation_ratio_w =
            ldg(rotation_ratio + rotation_offset_fhw + out_height * out_width);

          const scalar_t cur_input = deformable_im2col_bilinear(
              input + input_offset_temp,
              in_height,
              in_width,
              input_h,
              input_w);

          const scalar_t filter_h = f_h + rotation_ratio_h +
            (scope_height - filter_height) / 2.0;
          const scalar_t filter_w = f_w + rotation_ratio_w +
            (scope_width - filter_width) / 2.0;
          sum += cur_input * planar_bilinear(
              filter + filter_offset_temp,
              scope_height,
              scope_width,
              filter_h,
              filter_w);
        }
      }
    }
    output[thread_id] = sum;
  }
}

void DeformableSampleDepthwiseConv2dForward(
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor output) {
  int num_kernels = args.batch * args.channel * args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "DeformableSampleDepthwiseConv2dForward_GPU", ([&] {
        const scalar_t *input_ = input.data<scalar_t>();
        const scalar_t *offset_ = offset.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        const scalar_t *filter_ = filter.data<scalar_t>();
        scalar_t *output_ = output.data<scalar_t>();

        if (args.filter_height == 3 && args.filter_width == 3) {
          DeformableSampleDepthwiseConv2dForwardKernel<scalar_t, 3, 3>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels,
              input_,
              offset_,
              rotation_ratio_,
              filter_,
              args,
              output_);
        } else  {
          DeformableSampleDepthwiseConv2dForwardKernel<scalar_t, -1, -1>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels,
              input_,
              offset_,
              rotation_ratio_,
              filter_,
              args,
              output_);
        }

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in DeformableSampleDepthwiseConv2dForwardKernel: %s\n", cudaGetErrorString(err));
  }
}

template<typename scalar_t>
__global__ __launch_bounds__(1024, 2) void SampleDepthwiseConv2dBackwardDataKernel(
    int n,
    const scalar_t* out_grad,
    const scalar_t* rotation_ratio,
    const scalar_t* filter,
    const SampleDepthwiseArgs args,
    scalar_t* in_grad) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = args.filter_height;
  const int filter_width = args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    // Compute the indexes of this thread in the input.
    const int in_w = thread_id % in_width;
    const int in_h = (thread_id / in_width) % in_height;
    const int channel_idx = (thread_id / in_width / in_height) % channel;
    const int batch_idx = thread_id / channel / in_width / in_height;

    const int out_h_start = CudaMax<int>(
        0, (in_h + pad_height - (filter_height - 1) * dilation_height +
          stride_height - 1) / stride_height);
    const int out_h_end = CudaMin<int>(
        out_height - 1, (in_h + pad_height) / stride_height);
    const int out_w_start = CudaMax<int>(
        0, (in_w + pad_width - (filter_width - 1) * dilation_width +
          stride_width - 1) / stride_width);
    const int out_w_end = CudaMin<int>(
        out_width - 1, (in_w + pad_width) / stride_width);

    const int group_id = channel_idx % sampling_group;
    const int rotation_offset_temp =
      (batch_idx * sampling_group + group_id) *
      (filter_height * filter_width * 2) * out_height * out_width;
    const int filter_offset_temp = channel_idx * scope_height * scope_width;
    const int out_grad_offset_temp =
      (batch_idx * channel + channel_idx) * (out_height * out_width);

    scalar_t sum = 0.0f;
    for (int out_h = out_h_start; out_h <= out_h_end; ++out_h) {
      int f_h = in_h + pad_height - out_h * stride_height;

      if (f_h % dilation_height == 0) {
        f_h /= dilation_height;
        const int out_grad_offset_h = out_grad_offset_temp + out_h * out_width;

        for (int out_w = out_w_start; out_w <= out_w_end; ++out_w) {
          int f_w = in_w + pad_width - out_w * stride_width;

          if (f_w % dilation_width == 0) {
            f_w /= dilation_width;
            const int out_grad_offset = out_grad_offset_h + out_w;

            const int rotation_offset_fhw = rotation_offset_temp +
              (f_h * filter_width + f_w) * 2 * out_height * out_width +
              (out_h * out_width + out_w);
            const scalar_t rotation_ratio_h =
              ldg(rotation_ratio + rotation_offset_fhw);
            const scalar_t rotation_ratio_w =
              ldg(rotation_ratio + rotation_offset_fhw + out_height * out_width);

            const scalar_t filter_h = f_h + rotation_ratio_h +
              (scope_height - filter_height) / 2.0;
            const scalar_t filter_w = f_w + rotation_ratio_w +
              (scope_width - filter_width) / 2.0;
            sum += ldg(out_grad + out_grad_offset) * planar_bilinear(
                filter + filter_offset_temp,
                scope_height,
                scope_width,
                filter_h,
                filter_w);
          }
        }
      }
    }
    in_grad[thread_id] = sum;
  }
}

void SampleDepthwiseConv2dBackwardData(
    const at::Tensor out_grad,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor in_grad) {

  int num_kernels = args.batch * args.channel * args.in_height * args.in_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "SampleDepthwiseConv2dBackwardData_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        const scalar_t *filter_ = filter.data<scalar_t>();
        scalar_t *in_grad_ = in_grad.data<scalar_t>();

        SampleDepthwiseConv2dBackwardDataKernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            out_grad_,
            rotation_ratio_,
            filter_,
            args,
            in_grad_);

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SampleDepthwiseConv2dBackwardDataKernel: %s\n", cudaGetErrorString(err));
  }
}

template<typename scalar_t>
__global__ __launch_bounds__(1024, 2) void DeformableSampleDepthwiseConv2dBackwardDataKernel(
    int n,
    const scalar_t* out_backprop,
    const scalar_t* offset,
    const scalar_t* rotation_ratio,
    const scalar_t* filter,
    const SampleDepthwiseArgs args,
    scalar_t* in_grad) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = args.filter_height;
  const int filter_width = args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int in_c = (thread_id / out_width / out_height) % channel;
    // NOTE(Hang Gao @ 07/26): why feed data like this? -- because
    const int f_w = (thread_id / out_width / out_height / channel) % filter_width;
    const int f_h = (thread_id / out_width / out_height / channel / filter_width) % filter_height;
    const int out_b = (thread_id / out_width / out_height / channel / filter_width) / filter_height;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row = out_h * stride_height - pad_height + f_h * dilation_height;
    const int in_col = out_w * stride_width - pad_width + f_w * dilation_width;

    const int deformable_offset_temp =
      (out_b * (filter_height * filter_width) + (f_h * filter_width + f_w)) * 2 *
      out_height * out_width + (out_h * out_width + out_w);
    const int group_id = in_c % sampling_group;
    const int rotation_offset_temp =
      ((out_b * sampling_group + group_id) * (filter_height * filter_width) +
       (f_h * filter_width + f_w)) * 2 * out_height * out_width +
      (out_h * out_width + out_w);
    const scalar_t input_h = in_row + ldg(offset + deformable_offset_temp);
    const scalar_t input_w = in_col + ldg(
        offset + deformable_offset_temp + out_height * out_width);

    // Avoid repeated computation.
    if (input_h > -1 && input_w > -1 && input_h < in_height && input_w < in_width) {
      const int input_offset_temp = (out_b * channel + in_c) * (in_height * in_width);
      const int filter_offset_temp = in_c * scope_height * scope_width;
      const scalar_t out_bp = ldg(
          out_backprop +
          (out_b * channel + in_c) * (out_height * out_width) +
          (out_h * out_width + out_w));

      const scalar_t rotation_ratio_h = ldg(
          rotation_ratio + rotation_offset_temp);
      const scalar_t rotation_ratio_w = ldg(
          rotation_ratio + rotation_offset_temp + out_height * out_width);

      scalar_t cur_weight = 0;
      const scalar_t filter_h = f_h + rotation_ratio_h +
        (scope_height - filter_height) / 2.0;
      const scalar_t filter_w = f_w + rotation_ratio_w +
        (scope_width - filter_width) / 2.0;
      cur_weight = planar_bilinear(
          filter + filter_offset_temp,
          scope_height,
          scope_width,
          filter_h,
          filter_w);

      const scalar_t partial_sum = cur_weight * out_bp;
      deformable_im2col_bilinear_backward(
          partial_sum,
          input_h,
          input_w,
          in_height,
          in_width,
          in_grad + input_offset_temp);
    }
  }
}

void DeformableSampleDepthwiseConv2dBackwardData(
    const at::Tensor out_grad,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor in_grad) {

  int num_kernels = args.batch * args.filter_height * args.filter_width *
    args.channel * args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "DeformableSampleDepthwiseConv2dBackwardData_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *offset_ = offset.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        const scalar_t *filter_ = filter.data<scalar_t>();
        scalar_t *in_grad_ = in_grad.data<scalar_t>();

        DeformableSampleDepthwiseConv2dBackwardDataKernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            out_grad_,
            offset_,
            rotation_ratio_,
            filter_,
            args,
            in_grad_);

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "error in DeformableSampleDepthwiseConv2dBackwardDataKernel: %s\n",
        cudaGetErrorString(err));
  }
}

template <typename scalar_t, int kFilterWidth, int kFilterHeight>
__global__ __launch_bounds__(1024, 2) void SampleDepthwiseConv2dBackwardFilterKernel(
    int n,
    const scalar_t* out_backprop,
    const scalar_t* input,
    const scalar_t* rotation_ratio,
    const SampleDepthwiseArgs args,
    scalar_t* filter_backprop) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % channel;
    const int out_b = thread_id / out_width / out_height / channel;
    const int in_c = out_c;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row_start = out_h * stride_height - pad_height;
    const int in_col_start = out_w * stride_width - pad_width;
    const int in_row_end = in_row_start + (filter_height - 1) * dilation_height;
    const int in_col_end = in_col_start + (filter_width - 1) * dilation_width;

    const int input_offset_temp = (out_b * channel + in_c) * (in_height * in_width);
    const int group_id = in_c % sampling_group;
    const int rotation_offset_temp = (out_b * sampling_group + group_id) *
      (filter_height * filter_width * 2) * out_height * out_width +
      (out_h * out_width + out_w);
    const int filter_offset_temp = in_c * scope_height * scope_width;

    const scalar_t out_bp = ldg(out_backprop + thread_id);
    if (in_row_start >= 0 && in_col_start >= 0 &&
        in_row_end < in_height && in_col_end < in_width) {

      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_row = in_row_start + f_h * dilation_height;
        // Avoid repeated computation.
        const int input_offset_local = input_offset_temp + in_row * in_width;

        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_col = in_col_start + f_w * dilation_width;
          const int input_offset = input_offset_local + in_col;

          const int rotation_offset_fhw =
            rotation_offset_temp + (f_h * filter_width + f_w) * 2 * out_height * out_width;
          const scalar_t rotation_ratio_h = ldg(
              rotation_ratio + rotation_offset_fhw);
          const scalar_t rotation_ratio_w = ldg(
              rotation_ratio + rotation_offset_fhw + out_height * out_width);

          scalar_t partial_sum = ldg(input + input_offset) * out_bp;
          const scalar_t filter_h = f_h + rotation_ratio_h +
            (scope_height - filter_height) / 2.0;
          const scalar_t filter_w = f_w + rotation_ratio_w +
            (scope_width - filter_width) / 2.0;
          planar_bilinear_backward_data(
              partial_sum,
              scope_height,
              scope_width,
              filter_h,
              filter_w,
              filter_backprop + filter_offset_temp);
        }
      }
    } else {
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_row = in_row_start + f_h * dilation_height;
        // Avoid repeated computation.
        const int input_offset_local = input_offset_temp + in_row * in_width;

        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_col = in_col_start + f_w * dilation_width;;

          if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
            const int input_offset = input_offset_local + in_col;

            const int rotation_offset_fhw = rotation_offset_temp +
              (f_h * filter_width + f_w) * 2 * out_height * out_width;
            const scalar_t rotation_ratio_h = ldg(
                rotation_ratio + rotation_offset_fhw);
            const scalar_t rotation_ratio_w = ldg(
                rotation_ratio + rotation_offset_fhw + out_height * out_width);

            scalar_t partial_sum = ldg(input + input_offset) * out_bp;
            const scalar_t filter_h = f_h + rotation_ratio_h +
              (scope_height - filter_height) / 2.0;
            const scalar_t filter_w = f_w + rotation_ratio_w +
              (scope_width - filter_width) / 2.0;
            planar_bilinear_backward_data(
                partial_sum,
                scope_height,
                scope_width,
                filter_h,
                filter_w,
                filter_backprop + filter_offset_temp);
          }
        }
      }
    }
  }
}

void SampleDepthwiseConv2dBackwardFilter(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor rotation_ratio,
    const SampleDepthwiseArgs args,
    at::Tensor filter_grad) {

  int num_kernels = args.batch * args.channel * args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "SampleDepthwiseConv2dBackwardFilter_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *input_ = input.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        scalar_t *filter_grad_ = filter_grad.data<scalar_t>();

        if (args.filter_height == 3 && args.filter_width == 3) {
          SampleDepthwiseConv2dBackwardFilterKernel<scalar_t, 3, 3>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels,
              out_grad_,
              input_,
              rotation_ratio_,
              args,
              filter_grad_);
        } else  {
          SampleDepthwiseConv2dBackwardFilterKernel<scalar_t, -1, -1>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels,
              out_grad_,
              input_,
              rotation_ratio_,
              args,
              filter_grad_);
        }

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SampleDepthwiseConv2dBackwardFilterKernel: %s\n", cudaGetErrorString(err));
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename scalar_t, int kFilterWidth, int kFilterHeight>
__global__ __launch_bounds__(1024, 2) void DeformableSampleDepthwiseConv2dBackwardFilterKernel(
    int n,
    const scalar_t* out_backprop,
    const scalar_t* input,
    const scalar_t* offset,
    const scalar_t* rotation_ratio,
    const SampleDepthwiseArgs args,
    scalar_t* filter_backprop) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % channel;
    const int out_b = thread_id / out_width / out_height / channel;
    const int in_c = out_c;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row_start = out_h * stride_height - pad_height;
    const int in_col_start = out_w * stride_width - pad_width;

    const int input_offset_temp = (out_b * channel + in_c) * (in_height * in_width);
    const int deformation_offset_temp = out_b * (filter_height * filter_width * 2) *
      out_height * out_width + (out_h * out_width + out_w);
    const int group_id = in_c % sampling_group;
    const int rotation_offset_temp = (out_b * sampling_group + group_id) *
      (filter_height * filter_width * 2) * out_height * out_width +
      (out_h * out_width + out_w);
    const int filter_offset_temp = in_c * scope_height * scope_width;

    const scalar_t out_bp = ldg(out_backprop + thread_id);

    CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
      const int in_row = in_row_start + f_h * dilation_height;

      // Avoid repeated computation.
      CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
        const int in_col = in_col_start + f_w * dilation_width;

        const int deformation_offset_fhw = deformation_offset_temp +
          (f_h * filter_width + f_w) * 2 * out_height * out_width;
        const int rotation_offset_fhw = rotation_offset_temp +
          (f_h * filter_width + f_w) * 2 * out_height * out_width;
        const scalar_t input_h = in_row + ldg(
            offset + deformation_offset_fhw);
        const scalar_t input_w = in_col + ldg(
            offset + deformation_offset_fhw + out_height * out_width);

        if (input_h > -1 && input_w > -1 && input_h < in_height && input_w < in_width) {
          const scalar_t rotation_ratio_h = ldg(
              rotation_ratio + rotation_offset_fhw);
          const scalar_t rotation_ratio_w = ldg(
              rotation_ratio + rotation_offset_fhw + out_height * out_width);

          const scalar_t partial_sum = deformable_im2col_bilinear(
              input + input_offset_temp,
              in_height,
              in_width,
              input_h,
              input_w) * out_bp;
          const scalar_t filter_h = f_h + rotation_ratio_h +
            (scope_height - filter_height) / 2.0;
          const scalar_t filter_w = f_w + rotation_ratio_w +
            (scope_width - filter_width) / 2.0;
          planar_bilinear_backward_data(
              partial_sum,
              scope_height,
              scope_width,
              filter_h,
              filter_w,
              filter_backprop + filter_offset_temp);
        }
      }
    }
  }
}

void DeformableSampleDepthwiseConv2dBackwardFilter(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const SampleDepthwiseArgs args,
    at::Tensor filter_grad) {

  int num_kernels = args.batch * args.channel * args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "DeformableSampleDepthwiseConv2dBackwardFilter_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *input_ = input.data<scalar_t>();
        const scalar_t *offset_ = offset.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        scalar_t *filter_grad_ = filter_grad.data<scalar_t>();

        if (args.filter_height == 3 && args.filter_width == 3) {
          DeformableSampleDepthwiseConv2dBackwardFilterKernel<scalar_t, 3, 3>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels,
              out_grad_,
              input_,
              offset_,
              rotation_ratio_,
              args,
              filter_grad_);
        } else  {
          DeformableSampleDepthwiseConv2dBackwardFilterKernel<scalar_t, -1, -1>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels,
              out_grad_,
              input_,
              offset_,
              rotation_ratio_,
              args,
              filter_grad_);
        }

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in DeformableSampleDepthwiseConv2dBackwardFilterKernel: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(1024, 2) void DeformableSampleDepthwiseConv2dBackwardOffsetKernel(
    int n,
    const scalar_t* out_backprop,
    const scalar_t* input,
    const scalar_t* offset,
    const scalar_t* rotation_ratio,
    const scalar_t* filter,
    const SampleDepthwiseArgs args,
    scalar_t* offset_backprop) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = args.filter_height;
  const int filter_width = args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int bp_dir = (thread_id / out_width / out_height) % 2;
    const int f_w = (thread_id / out_width / out_height / 2) % filter_width;
    const int f_h = (thread_id / out_width / out_height / 2 / filter_width) % filter_height;
    const int out_b = (thread_id / out_width / out_height / 2 / filter_width) / filter_height;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row = out_h * stride_height - pad_height + f_h * dilation_height;
    const int in_col = out_w * stride_width - pad_width + f_w * dilation_width;

    const int deformable_offset_temp =
      (out_b * (filter_height * filter_width) + (f_h * filter_width + f_w)) * 2 *
      out_height * out_width +
      (out_h * out_width + out_w);
    const scalar_t input_h = in_row + ldg(
        offset + deformable_offset_temp);
    const scalar_t input_w = in_col + ldg(
        offset + deformable_offset_temp + out_height * out_width);

    scalar_t coord_gradient = 0;
    // Avoid repeated computation.
    if (input_h > -1 && input_w > -1 && input_h < in_height && input_w < in_width) {

      for (int in_c = 0; in_c < channel; in_c++) {
        const int group_id = in_c % sampling_group;
        const int rotation_offset_temp = ((out_b * sampling_group + group_id) *
            (filter_height * filter_width) + (f_h * filter_width + f_w)) * 2 *
          out_height * out_width + (out_h * out_width + out_w);
        const scalar_t rotation_ratio_h = ldg(
            rotation_ratio + rotation_offset_temp);
        const scalar_t rotation_ratio_w = ldg(
            rotation_ratio + rotation_offset_temp + out_height * out_width);
        scalar_t filter_h = f_h + rotation_ratio_h +
          (scope_height - filter_height) / 2.0;
        scalar_t filter_w = f_w + rotation_ratio_w +
          (scope_width - filter_width) / 2.0;

          const int input_offset_temp = (out_b * channel + in_c) * (in_height * in_width);
          const int filter_offset_temp = in_c * scope_height * scope_width;
          const scalar_t out_bp = ldg(
              out_backprop +
              (out_b * channel + in_c) * (out_height * out_width) +
              (out_h * out_width + out_w));

        scalar_t cur_weight = planar_bilinear(
            filter + filter_offset_temp,
            scope_height,
            scope_width,
            filter_h,
            filter_w);
        scalar_t partial_sum = cur_weight * out_bp;
        coord_gradient += get_coordinate_weight(
            input_h,
            input_w,
            in_height,
            in_width,
            input + input_offset_temp,
            bp_dir) * partial_sum;
      }
    }

    offset_backprop[thread_id] = coord_gradient;
  }
}

void DeformableSampleDepthwiseConv2dBackwardOffset(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor offset_grad) {

  int num_kernels = args.batch * args.filter_height * args.filter_width * 2 *
    args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "DeformableSampleDepthwiseConv2dBackwardOffset_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *input_ = input.data<scalar_t>();
        const scalar_t *offset_ = offset.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        const scalar_t *filter_ = filter.data<scalar_t>();
        scalar_t *offset_grad_ = offset_grad.data<scalar_t>();

        DeformableSampleDepthwiseConv2dBackwardOffsetKernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            out_grad_,
            input_,
            offset_,
            rotation_ratio_,
            filter_,
            args,
            offset_grad_);

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in DeformableSampleDepthwiseConv2dBackwardOffsetKernel: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(1024, 2) void SampleDepthwiseConv2dBackwardRotationKernel(
    int n,
    const scalar_t* out_backprop,
    const scalar_t* input,
    const scalar_t* rotation_ratio,
    const scalar_t* filter,
    const SampleDepthwiseArgs args,
    scalar_t* rotation_backprop) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = args.filter_height;
  const int filter_width = args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int bp_dir = (thread_id / out_width / out_height) % 2;
    const int f_w = (thread_id / out_width / out_height / 2) % filter_width;
    const int f_h = (thread_id / out_width / out_height / 2 / filter_width) % filter_height;
    const int group_id = (thread_id / out_width / out_height / 2 /
        filter_width / filter_height) % sampling_group;
    const int out_b = (thread_id / out_width / out_height / 2 /
        filter_width / filter_height) / sampling_group;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row = out_h * stride_height - pad_height + f_h * dilation_height;
    const int in_col = out_w * stride_width - pad_width + f_w * dilation_width;

    const int rotation_offset_temp =
      ((out_b * sampling_group + group_id) * (filter_height * filter_width) +
       (f_h * filter_width + f_w)) * 2 * out_height * out_width +
      (out_h * out_width + out_w);
    const scalar_t rotation_ratio_h = ldg(
        rotation_ratio + rotation_offset_temp);
    const scalar_t rotation_ratio_w = ldg(
        rotation_ratio + rotation_offset_temp + out_height * out_width);
    scalar_t filter_h = f_h + rotation_ratio_h +
      (scope_height - filter_height) / 2.0;
    scalar_t filter_w = f_w + rotation_ratio_w +
      (scope_width - filter_width) / 2.0;

    scalar_t coord_gradient = 0;
    // Avoid repeated computation.
    if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
      for (int in_c = group_id; in_c < channel; in_c += sampling_group) {
        const int input_offset_temp =
          (out_b * channel + in_c) * (in_height * in_width) +
          (in_row * in_width + in_col);
        const int filter_offset_temp = in_c * scope_height * scope_width;
        const scalar_t out_bp = ldg(
            out_backprop +
            (out_b * channel + in_c) * (out_height * out_width) +
            (out_h * out_width + out_w));

        scalar_t partial_sum = ldg(input + input_offset_temp) * out_bp;
        coord_gradient += planar_bilinear_backward_coord(
            partial_sum,
            filter + filter_offset_temp,
            scope_height,
            scope_width,
            filter_h,
            filter_w,
            bp_dir);
      }
    }

    rotation_backprop[thread_id] = coord_gradient;
  }
}

void SampleDepthwiseConv2dBackwardRotation(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor rotation_grad) {

  int num_kernels = args.batch *
    args.sampling_group * args.filter_height * args.filter_width * 2 *
    args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "SampleDepthwiseConv2dBackwardRotation_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *input_ = input.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        const scalar_t *filter_ = filter.data<scalar_t>();
        scalar_t *rotation_grad_ = rotation_grad.data<scalar_t>();

        SampleDepthwiseConv2dBackwardRotationKernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            out_grad_,
            input_,
            rotation_ratio_,
            filter_,
            args,
            rotation_grad_);

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SampleDepthwiseConv2dBackwardRotationKernel: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(1024, 2) void DeformableSampleDepthwiseConv2dBackwardRotationKernel(
    int n,
    const scalar_t* out_backprop,
    const scalar_t* input,
    const scalar_t* offset,
    const scalar_t* rotation_ratio,
    const scalar_t* filter,
    const SampleDepthwiseArgs args,
    scalar_t* rotation_backprop) {

  const int channel = args.channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = args.filter_height;
  const int filter_width = args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int dilation_height = args.dilation_height;
  const int dilation_width = args.dilation_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int scope_height = args.scope_height;
  const int scope_width = args.scope_width;
  const int sampling_group = args.sampling_group;

  CUDA_KERNEL_LOOP(thread_id, n) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int bp_dir = (thread_id / out_width / out_height) % 2;
    const int f_w = (thread_id / out_width / out_height / 2) % filter_width;
    const int f_h = (thread_id / out_width / out_height / 2 / filter_width) % filter_height;
    const int group_id = (thread_id / out_width / out_height / 2 /
        filter_width / filter_height) % sampling_group;
    const int out_b = (thread_id / out_width / out_height / 2 /
        filter_width / filter_height) / sampling_group;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row = out_h * stride_height - pad_height + f_h * dilation_height;
    const int in_col = out_w * stride_width - pad_width + f_w * dilation_width;

    const int deformable_offset_temp =
      (out_b * (filter_height * filter_width) + (f_h * filter_width + f_w)) * 2 *
      out_height * out_width + (out_h * out_width + out_w);
    const int rotation_offset_temp =
      ((out_b * sampling_group + group_id) * (filter_height * filter_width) +
       (f_h * filter_width + f_w)) * 2 * out_height * out_width +
      (out_h * out_width + out_w);
    const scalar_t input_h = in_row + ldg(
        offset + deformable_offset_temp);
    const scalar_t input_w = in_col + ldg(
        offset + deformable_offset_temp + out_height * out_width);

    scalar_t coord_gradient = 0;
    // Avoid repeated computation.
    if (input_h > -1 && input_w > -1 && input_h < in_height && input_w < in_width) {
      const scalar_t rotation_ratio_h = ldg(
          rotation_ratio + rotation_offset_temp);
      const scalar_t rotation_ratio_w = ldg(
          rotation_ratio + rotation_offset_temp + out_height * out_width);
      scalar_t filter_h = f_h + rotation_ratio_h +
        (scope_height - filter_height) / 2.0;
      scalar_t filter_w = f_w + rotation_ratio_w +
        (scope_width - filter_width) / 2.0;

      for (int in_c = group_id; in_c < channel; in_c += sampling_group) {
        const int input_offset_temp = (out_b * channel + in_c) * (in_height * in_width);
        const int filter_offset_temp = in_c * scope_height * scope_width;
        const scalar_t out_bp = ldg(
            out_backprop +
            (out_b * channel + in_c) * (out_height * out_width) +
            (out_h * out_width + out_w));

        scalar_t partial_sum = deformable_im2col_bilinear(
            input + input_offset_temp,
            in_height,
            in_width,
            input_h,
            input_w) * out_bp;
        coord_gradient += planar_bilinear_backward_coord(
            partial_sum,
            filter + filter_offset_temp,
            scope_height,
            scope_width,
            filter_h,
            filter_w,
            bp_dir);
      }
    }

    rotation_backprop[thread_id] = coord_gradient;
  }
}

void DeformableSampleDepthwiseConv2dBackwardRotation(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor rotation_grad) {

  int num_kernels = args.batch *
    args.sampling_group * args.filter_height * args.filter_width * 2 *
    args.out_height * args.out_width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.type(), "DeformableSampleDepthwiseConv2dBackwardRotation_GPU", ([&] {
        const scalar_t *out_grad_ = out_grad.data<scalar_t>();
        const scalar_t *input_ = input.data<scalar_t>();
        const scalar_t *offset_ = offset.data<scalar_t>();
        const scalar_t *rotation_ratio_ = rotation_ratio.data<scalar_t>();
        const scalar_t *filter_ = filter.data<scalar_t>();
        scalar_t *rotation_grad_ = rotation_grad.data<scalar_t>();

        DeformableSampleDepthwiseConv2dBackwardRotationKernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            out_grad_,
            input_,
            offset_,
            rotation_ratio_,
            filter_,
            args,
            rotation_grad_);

      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in DeformableSampleDepthwiseConv2dBackwardRotationKernel: %s\n", cudaGetErrorString(err));
  }
}
