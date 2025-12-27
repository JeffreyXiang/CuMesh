from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

if not IS_HIP:
    cc_flag = []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

setup(
    name="cumesh",
    packages=[
        'cumesh',
    ],
    ext_modules=[
        CUDAExtension(
            name="cumesh._C",
            sources=[
                # Hashmap functions
                "src/hash/hash.cu",
                
                # CuMesh
                "src/atlas.cu",
                "src/clean_up.cu",
                "src/cumesh.cu",
                "src/connectivity.cu",
                "src/geometry.cu",
                "src/io.cu",
                "src/simplify.cu",
                "src/shared.cu",
                
                # Remeshing
                "src/remesh/simple_dual_contour.cu",
                "src/remesh/svox2vert.cu",
                
                # main
                "src/ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3","-std=c++17"] + cc_flag,
            }
        ),
        CUDAExtension(
            name='cumesh._cubvh',
            sources=[
                'third_party/cubvh/src/bvh.cu',
                'third_party/cubvh/src/api_gpu.cu',
                'third_party/cubvh/src/bindings.cpp',
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/cubvh/include"),
                os.path.join(ROOT, "third_party/cubvh/third_party/eigen"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3","-std=c++17"] + cc_flag + [
                    "--extended-lambda",
                    "--expt-relaxed-constexpr",
                    # The following definitions must be undefined
                    # since we need half-precision operation.
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ]
            }
        ),
        CUDAExtension(
            name='cumesh._cumesh_xatlas',
            sources=[
                'third_party/xatlas/xatlas.cpp',
                'third_party/xatlas/binding.cpp',
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)