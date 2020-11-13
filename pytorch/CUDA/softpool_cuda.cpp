#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

int SoftPool1dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int dim,
                              const int kernel_d, const int stride_d,
                              at::Tensor output);

int SoftPool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int dim, const int kernel_d,
                               const int stride_d, at::Tensor input_grad);

int SoftPool2dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int height,
                              const int width, const int kernel_h,
                              const int kernel_w, const int stride_h,
                              const int stride_w, at::Tensor output);

int SoftPool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w,
                               at::Tensor input_grad);

int SoftPool3dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int depth,
                              const int height, const int width,
                              const int kernel_d, const int kernel_h,
                              const int kernel_w, const int stride_d,
                              const int stride_h, const int stride_w,
                              at::Tensor output);

int SoftPool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int depth, const int height,
                               const int width, const int kernel_d,
                               const int kernel_h, const int kernel_w,
                               const int stride_d, const int stride_h,
                               const int stride_w, at::Tensor input_grad);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

int softpool1d_forward_cuda(at::Tensor input, const std::tuple<int> kernel,
                            const std::tuple<int> stride, at::Tensor output) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    int batches = input.size(0);
    int channels = input.size(1);
    int dim = input.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    SoftPool1dForwardLauncher(input, batches,
                              channels, dim,
                              kernel_d, stride_d,
                              output);
    return 1;
}

int softpool1d_backward_cuda(const at::Tensor output_grad, const at::Tensor input,
                             const std::tuple<int> kernel, const std::tuple<int> stride,
                             at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);


    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int dim = input_grad.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    SoftPool1dBackwardLauncher(output_grad, input,
                               batches, channels,
                               dim, kernel_d,
                               stride_d, input_grad);
    return 1;
}


int softpool2d_forward_cuda(at::Tensor input, const std::tuple<int, int> kernel,
                            const std::tuple<int, int> stride, at::Tensor output) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    int batches = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    SoftPool2dForwardLauncher(input, batches,
                              channels, height,
                              width, kernel_h,
                              kernel_w, stride_h,
                              stride_w, output);
    return 1;
}

int softpool2d_backward_cuda(const at::Tensor output_grad, const at::Tensor input,
                             const std::tuple<int, int> kernel, const std::tuple<int, int> stride,
                             at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);


    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int height = input_grad.size(2);
    int width = input_grad.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    SoftPool2dBackwardLauncher(output_grad, input,
                               batches, channels,
                               height, width,
                               kernel_h, kernel_w,
                               stride_h, stride_w,
                               input_grad);
    return 1;
}


int softpool3d_forward_cuda(at::Tensor input, const std::tuple<int, int, int> kernel,
                            const std::tuple<int, int, int> stride, at::Tensor output) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    int batches = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    SoftPool3dForwardLauncher(input, batches,
                              channels, depth,
                              height, width,
                              kernel_d, kernel_h,
                              kernel_w, stride_d,
                              stride_h, stride_w,
                              output);
    return 1;
}

int softpool3d_backward_cuda(const at::Tensor output_grad, const at::Tensor input,
                             const std::tuple<int, int, int> kernel, const std::tuple<int, int, int> stride,
                             at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);


    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int depth = input_grad.size(2);
    int height = input_grad.size(3);
    int width = input_grad.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    SoftPool3dBackwardLauncher(output_grad, input,
                               batches, channels,
                               depth, height,
                               width, kernel_d,
                               kernel_h, kernel_w,
                               stride_d, stride_h,
                               stride_w, input_grad);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_1d", &softpool1d_forward_cuda, "SoftPool1d forward (CUDA)");
  m.def("backward_1d", &softpool1d_backward_cuda, "SoftPool1d backward (CUDA)");
  m.def("forward_2d", &softpool2d_forward_cuda, "SoftPool2d forward (CUDA)");
  m.def("backward_2d", &softpool2d_backward_cuda, "SoftPool2d backward (CUDA)");
  m.def("forward_3d", &softpool3d_forward_cuda, "SoftPool3d forward (CUDA)");
  m.def("backward_3d", &softpool3d_backward_cuda, "SoftPool3d backward (CUDA)");
}
