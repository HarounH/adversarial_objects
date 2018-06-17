import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

rasterize = load(
        'rasterize', [os.path.join(dir_path,'rasterize_cuda.cpp'),
                          os.path.join(dir_path, 'rasterize_cuda_kernel.cu')],
        verbose=True, extra_cuda_cflags=['-arch=sm_60', '-gencode=arch=compute_60,code=sm_60'])
help(rasterize)

class RasterizeFunction(Function):

    @staticmethod
    def forward(ctx):
        # face_index_map.forward()
        return

    @staticmethod
    def backward(ctx):
        # face_index_map.backward()
        return
