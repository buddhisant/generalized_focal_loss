import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extensions = [CUDAExtension(name="cuda_tools.ops",
                            sources=glob.glob("cuda_tools/ops/src/*"),
                            include_dirs=[os.path.abspath("cuda_tools/ops")]), ]

setup(
    name='cuda_tools',
    version="1.0.0",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
)
