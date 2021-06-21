from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='SoftPool',
    version='1.1',
    description='CUDA-accelerated package for performing 1D/2D/3D SoftPool',
    author='Alexandros Stergiou',
    author_email='alexstergiou5@gmail.com',
    license='MIT',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('softpool_cuda', [
            'CUDA/softpool_cuda.cpp',
            'CUDA/softpool_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })
