#include <torch/torch.h>

// CUDA forward declarations

at::Tensor load_textures_cuda(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor is_update,
        at::Tensor textures);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor load_textures(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor is_update,
        at::Tensor textures) {

    CHECK_INPUT(image);
    CHECK_INPUT(faces);
    CHECK_INPUT(is_update);
    CHECK_INPUT(textures);

    return load_textures_cuda(image, faces, is_update, textures);
                                      
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_textures", &load_textures_cuda, "LOAD_TEXTURES (CUDA)");
}
