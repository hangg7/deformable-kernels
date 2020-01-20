#include <torch/extension.h>

#include <cmath>
#include <vector>

#include "nd_linear_sample_cuda.h"

/*------------------------------------------------------------------------------------------------------------*/

int nd_linear_sample_forward_cuda(at::Tensor data, at::Tensor shape, at::Tensor coord, at::Tensor output) {
  data = data.contiguous();
  shape = shape.contiguous();
  coord = coord.contiguous();

  SampleArgs args;
  args.batch = coord.size(0);
  args.channel = data.size(0);
  args.spatial_dims = data.dim() - 1;
  args.prod_shape = data.stride(0);

  output = output.view({args.batch, args.channel});

  NdLinearSampleForward(data, shape, coord, args, output);

  return 1;
}


int nd_linear_sample_backward_data_cuda(at::Tensor out_grad, at::Tensor data, at::Tensor shape, at::Tensor coord, at::Tensor in_grad) {
  out_grad = out_grad.contiguous();
  shape = shape.contiguous();
  coord = coord.contiguous();

  SampleArgs args;
  args.batch = coord.size(0);
  args.channel = data.size(0);
  args.spatial_dims = data.dim() - 1;
  args.prod_shape = data.stride(0);

  in_grad = in_grad.view_as(data).zero_();
  
  NdLinearSampleBackwardData(out_grad, shape, coord, args, in_grad);
      
  return 1;
}

int nd_linear_sample_backward_coord_cuda(at::Tensor out_grad, at::Tensor data, at::Tensor shape, at::Tensor coord, at::Tensor coord_grad_c) {
  out_grad = out_grad.contiguous();
  data = data.contiguous();
  shape = shape.contiguous();
  coord = coord.contiguous();

  SampleArgs args;
  args.batch = coord.size(0);
  args.channel = data.size(0);
  args.spatial_dims = data.dim() - 1;
  args.prod_shape = data.stride(0);

  coord_grad_c = coord_grad_c.view({args.batch, args.spatial_dims, args.channel}).zero_();

  NdLinearSampleBackwardCoord(out_grad, data, shape, coord, args, coord_grad_c);
      
  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nd_linear_sample_forward_cuda", &nd_linear_sample_forward_cuda,
        "nd_linear_sample forward (CUDA)");
  m.def("nd_linear_sample_backward_data_cuda", &nd_linear_sample_backward_data_cuda,
        "nd_linear_sample_backward_data (CUDA)");
  m.def("nd_linear_sample_backward_coord_cuda", &nd_linear_sample_backward_coord_cuda,
        "nd_linear_sample_backward_coord (CUDA)");
}
