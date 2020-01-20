struct DepthwiseArgs {
  // Input layer dimensions
  int batch;
  int channel;
  int in_height;
  int in_width;

  // Weight layer dimensions
  int filter_height;
  int filter_width;
  int stride_height;
  int stride_width;
  int pad_height;
  int pad_width;
  int dilation_height;
  int dilation_width;

  // Output layer dimensions
  int out_height;
  int out_width;
};

struct SampleDepthwiseArgs : DepthwiseArgs {
  // Weight layer dimensions
  int scope_height;
  int scope_width;
  int sampling_group;
};

void SampleDepthwiseConv2dForward(
    const at::Tensor input,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor output);

void DeformableSampleDepthwiseConv2dForward(
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor output);

void SampleDepthwiseConv2dBackwardData(
    const at::Tensor out_grad,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor in_grad);

void DeformableSampleDepthwiseConv2dBackwardData(
    const at::Tensor out_grad,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor in_grad);

void SampleDepthwiseConv2dBackwardFilter(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor rotation_ratio,
    const SampleDepthwiseArgs args,
    at::Tensor filter_grad);

void DeformableSampleDepthwiseConv2dBackwardFilter(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const SampleDepthwiseArgs args,
    at::Tensor filter_grad);

void DeformableSampleDepthwiseConv2dBackwardOffset(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor offset_grad);

void SampleDepthwiseConv2dBackwardRotation(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor rotation_grad);

void DeformableSampleDepthwiseConv2dBackwardRotation(
    const at::Tensor out_grad,
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor rotation_ratio,
    const at::Tensor filter,
    const SampleDepthwiseArgs args,
    at::Tensor rotation_grad);
