from setuptools import setup, find_packages
import unittest

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = ['-gencode=arch=compute_30,code=sm_30',
              '-gencode=arch=compute_35,code=sm_35',
              '-gencode=arch=compute_50,code=sm_50',
              '-gencode=arch=compute_52,code=sm_52',
              '-gencode=arch=compute_60,code=sm_60',
              '-gencode=arch=compute_61,code=sm_61',
              '-gencode=arch=compute_70,code=sm_70',
              '-gencode=arch=compute_70,code=compute_70']

def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

ext_modules=[
    CUDAExtension('neural_renderer.cuda.load_textures', [
        'neural_renderer/cuda/load_textures_cuda.cpp',
        'neural_renderer/cuda/load_textures_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': CUDA_FLAGS}),
    CUDAExtension('neural_renderer.cuda.rasterize', [
        'neural_renderer/cuda/rasterize_cuda.cpp',
        'neural_renderer/cuda/rasterize_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': CUDA_FLAGS}),
    CUDAExtension('neural_renderer.cuda.create_texture_image', [
        'neural_renderer/cuda/create_texture_image_cuda.cpp',
        'neural_renderer/cuda/create_texture_image_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': CUDA_FLAGS}),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "A 3D mesh renderer for neural networks"',
    author='Nikolaos Kolotouros',
    author_email='nkolot@seas.upenn.edu',
    license='MIT License',
    version='1.1.3',
    name='neural_renderer',
    test_suite='setup.test_all',
    packages=find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
