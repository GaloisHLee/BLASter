from pathlib import Path
from sys import argv, platform
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

import numpy as np


# Make sure OpenMP is used in Cython and Eigen.
include_dirs = [np.get_include()]

# Configure OpenMP flags per platform/compiler
if platform.startswith("win"):
    openmp_compile_args = ['/openmp']
    openmp_link_args = []
elif platform == 'darwin':
    # macOS clang needs the preprocessor flag and links against libomp
    # Requires libomp installed (e.g., `brew install libomp` or `conda install -c conda-forge llvm-openmp`)
    openmp_compile_args = ['-Xpreprocessor', '-fopenmp']
    openmp_link_args = ['-lomp']
else:
    # GCC/LLVM on Linux
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']

# Look for the Eigen library in `/usr/include` and `~/.local/include`.
for f in [Path('eigen3'), Path('/usr/include/eigen3'), Path.home().joinpath('.local/include/eigen3'),
          Path('/opt/homebrew/include/eigen3'), Path('/usr/local/include/eigen3')]:
    if f.exists() and f.is_dir():
        include_dirs += [str(f)]
        break
else:
    print("ERROR: Eigen3 library is required!")
    print("NOTE : Please run 'make eigen3'")
    exit(1)

# Compile with extra arguments
compile_args = [
    '/std:c++17' if platform.startswith("win") else '-std=c++17',
    '/DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'
    if platform.startswith("win") else '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION',
]
compile_args += openmp_compile_args

# Link with extra arguments
link_args = []
link_args += openmp_link_args

if '--cython-gdb' in argv:
    # Debug arguments
    debug_args = [
        '-O1',
        '-fsanitize=address,undefined',
        '-g',
        '-fno-omit-frame-pointer',
    ]
    compile_args += debug_args
    link_args += debug_args
else:
    # "Release" arguments
    compile_args += (
        ['/O2', '/DEIGEN_NO_DEBUG']
        if platform.startswith("win")
        else ['-O3', '-march=native', '-DEIGEN_NO_DEBUG']
    )

extensions = [Extension(
    name="blaster._core",
    sources=["core/blaster.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args
)]

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, language_level="3", build_dir='build/cpp'),
)
