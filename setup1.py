from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension


setup(
    name='my_lib',
    version='0.0',
    description='Learning setup',
    packages=find_packages(),
    ext_package='trt_pose',
    ext_modules=[cpp_extension.CppExtension('plugins', [
        'Learn_cpp/learn.cpp',
    ])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
    ],
)
