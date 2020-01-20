#include <torch/extension.h>

#include <cmath>
#include <vector>

#include "filter_sample_depthwise_cuda.h"

int sample_depthwise_forward_cuda(
    at::Tensor input,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor output,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  input = input.contiguous();
  rotation = rotation.contiguous();
  filter = filter.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  output = output.view({
      sdw_args.batch,
      sdw_args.channel,
      sdw_args.out_height,
      sdw_args.out_width});

  SampleDepthwiseConv2dForward(
      input,
      rotation,
      filter,
      sdw_args,
      output);

  return 1;
}

int sample_depthwise_backward_data_cuda(
    at::Tensor gradOutput,
    at::Tensor input,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor gradInput,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  gradOutput = gradOutput.contiguous();
  rotation = rotation.contiguous();
  filter = filter.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  gradInput = gradInput.view({
      sdw_args.batch,
      sdw_args.channel,
      sdw_args.in_height,
      sdw_args.in_width});

  SampleDepthwiseConv2dBackwardData(
      gradOutput,
      rotation,
      filter,
      sdw_args,
      gradInput);

  return 1;
}

int sample_depthwise_backward_filter_cuda(
    at::Tensor gradOutput,
    at::Tensor input,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor gradFilter,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  gradOutput = gradOutput.contiguous();
  input = input.contiguous();
  rotation = rotation.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  gradFilter = gradFilter.view({
      sdw_args.channel,
      1,
      sdw_args.scope_height,
      sdw_args.scope_width});

  SampleDepthwiseConv2dBackwardFilter(
      gradOutput,
      input,
      rotation,
      sdw_args,
      gradFilter);

  return 1;
}

int sample_depthwise_backward_rotation_cuda(
    at::Tensor gradOutput,
    at::Tensor input,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor gradRotation,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  gradOutput = gradOutput.contiguous();
  input = input.contiguous();
  rotation = rotation.contiguous();
  filter = filter.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  gradRotation = gradRotation.view({
      sdw_args.batch,
      groupS * kH * kW * 2,
      sdw_args.out_height,
      sdw_args.out_width});

  SampleDepthwiseConv2dBackwardRotation(
      gradOutput,
      input,
      rotation,
      filter,
      sdw_args,
      gradRotation);

  return 1;
}

int deformable_sample_depthwise_forward_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor output,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  input = input.contiguous();
  offset = offset.contiguous();
  rotation = rotation.contiguous();
  filter = filter.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  output = output.view({
      sdw_args.batch,
      sdw_args.channel,
      sdw_args.out_height,
      sdw_args.out_width});

  DeformableSampleDepthwiseConv2dForward(
      input,
      offset,
      rotation,
      filter,
      sdw_args,
      output);

  return 1;
}

int deformable_sample_depthwise_backward_data_cuda(
    at::Tensor gradOutput,
    at::Tensor input,
    at::Tensor offset,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor gradInput,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  gradOutput = gradOutput.contiguous();
  offset = offset.contiguous();
  rotation = rotation.contiguous();
  filter = filter.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  gradInput = gradInput.view({
      sdw_args.batch,
      sdw_args.channel,
      sdw_args.in_height,
      sdw_args.in_width});

  DeformableSampleDepthwiseConv2dBackwardData(
      gradOutput,
      offset,
      rotation,
      filter,
      sdw_args,
      gradInput);

  return 1;
}

int deformable_sample_depthwise_backward_filter_cuda(
    at::Tensor gradOutput,
    at::Tensor input,
    at::Tensor offset,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor gradFilter,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  gradOutput = gradOutput.contiguous();
  input = input.contiguous();
  offset = offset.contiguous();
  rotation = rotation.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  gradFilter = gradFilter.view({
      sdw_args.channel,
      1,
      sdw_args.scope_height,
      sdw_args.scope_width});

  DeformableSampleDepthwiseConv2dBackwardFilter(
      gradOutput,
      input,
      offset,
      rotation,
      sdw_args,
      gradFilter);

  return 1;
}

int deformable_sample_depthwise_backward_offset_cuda(
    at::Tensor gradOutput,
    at::Tensor input,
    at::Tensor offset,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor gradOffset,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  gradOutput = gradOutput.contiguous();
  input = input.contiguous();
  offset = offset.contiguous();
  rotation = rotation.contiguous();
  filter = filter.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  gradOffset = gradOffset.view({
      sdw_args.batch,
      kH * kW * 2,
      sdw_args.out_height,
      sdw_args.out_width});

  DeformableSampleDepthwiseConv2dBackwardOffset(
      gradOutput,
      input,
      offset,
      rotation,
      filter,
      sdw_args,
      gradOffset);

  return 1;
}

int deformable_sample_depthwise_backward_rotation_cuda(
    at::Tensor gradOutput,
    at::Tensor input,
    at::Tensor offset,
    at::Tensor rotation,
    at::Tensor filter,
    at::Tensor gradRotation,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int scopeH,
    int scopeW,
    int groupS) {

  gradOutput = gradOutput.contiguous();
  input = input.contiguous();
  offset = offset.contiguous();
  rotation = rotation.contiguous();
  filter = filter.contiguous();

  SampleDepthwiseArgs sdw_args;
  sdw_args.batch = input.size(0);
  sdw_args.channel = input.size(1);
  sdw_args.in_height = input.size(2);
  sdw_args.in_width = input.size(3);
  sdw_args.filter_height = kH;
  sdw_args.filter_width = kW;
  sdw_args.stride_height = dH;
  sdw_args.stride_width = dW;
  sdw_args.pad_height = padH;
  sdw_args.pad_width = padW;
  sdw_args.dilation_height = dilationH;
  sdw_args.dilation_width = dilationW;
  sdw_args.out_height = (sdw_args.in_height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  sdw_args.out_width = (sdw_args.in_width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  sdw_args.scope_height = scopeH;
  sdw_args.scope_width = scopeW;
  sdw_args.sampling_group = groupS;

  gradRotation = gradRotation.view({
      sdw_args.batch,
      groupS * kH * kW * 2,
      sdw_args.out_height,
      sdw_args.out_width});

  DeformableSampleDepthwiseConv2dBackwardRotation(
      gradOutput,
      input,
      offset,
      rotation,
      filter,
      sdw_args,
      gradRotation);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_depthwise_forward_cuda",
      &sample_depthwise_forward_cuda,
      "sample_depthwise_forward (CUDA)");
  m.def("sample_depthwise_backward_data_cuda",
      &sample_depthwise_backward_data_cuda,
      "sample_depthwise_backward_data (CUDA)");
  m.def("sample_depthwise_backward_filter_cuda",
      &sample_depthwise_backward_filter_cuda,
      "sample_depthwise_backward_filter (CUDA)");
  m.def("sample_depthwise_backward_rotation_cuda",
      &sample_depthwise_backward_rotation_cuda,
      "sample_depthwise_backward_rotation (CUDA)");

  m.def("deformable_sample_depthwise_forward_cuda",
      &deformable_sample_depthwise_forward_cuda,
      "deformable_sample_depthwise_forward (CUDA)");
  m.def("deformable_sample_depthwise_backward_data_cuda",
      &deformable_sample_depthwise_backward_data_cuda,
      "deformable_sample_depthwise_backward_data (CUDA)");
  m.def("deformable_sample_depthwise_backward_filter_cuda",
      &deformable_sample_depthwise_backward_filter_cuda,
      "deformable_sample_depthwise_backward_filter (CUDA)");
  m.def("deformable_sample_depthwise_backward_offset_cuda",
      &deformable_sample_depthwise_backward_offset_cuda,
      "deformable_sample_depthwise_backward_offset (CUDA)");
  m.def("deformable_sample_depthwise_backward_rotation_cuda",
      &deformable_sample_depthwise_backward_rotation_cuda,
      "deformable_sample_depthwise_backward_rotation (CUDA)");
}
