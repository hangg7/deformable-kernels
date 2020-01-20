struct SampleArgs {
  // Input layer dimensions
  int batch;
  int channel;
  int spatial_dims;
  int prod_shape;
};

void NdLinearSampleForward(const at::Tensor data, const at::Tensor shape, const at::Tensor coord, const SampleArgs args, at::Tensor output);

void NdLinearSampleBackwardData(const at::Tensor out_grad, const at::Tensor shape, const at::Tensor coord, const SampleArgs args, at::Tensor in_grad);

void NdLinearSampleBackwardCoord(const at::Tensor out_grad, const at::Tensor data, const at::Tensor shape, const at::Tensor coord, const SampleArgs args, at::Tensor coord_grad_c);