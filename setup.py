from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
IS_WINDOWS = platform.system() == "Windows"

# -------------------------------------------------
# Detect backend
# -------------------------------------------------
if BUILD_TARGET == "auto":
    IS_HIP = bool(IS_HIP_EXTENSION)
elif BUILD_TARGET == "cuda":
    IS_HIP = False
elif BUILD_TARGET == "rocm":
    IS_HIP = True
else:
    raise ValueError(f"Invalid BUILD_TARGET={BUILD_TARGET}")

# -------------------------------------------------
# Common flags
# -------------------------------------------------
cxx_flags = []
nvcc_flags = []

if IS_WINDOWS:
    # Required for MSVC + nvcc + torch headers
    cxx_flags += [
        "/O2",
        "/std:c++17",
        "/EHsc", 
        "/permissive-", 
        "/Zc:__cplusplus"
    ]
    nvcc_flags += [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-Xcompiler=/std:c++17",
        "-Xcompiler=/EHsc", 
        "-Xcompiler=/permissive-", 
        "-Xcompiler=/Zc:__cplusplus"
    ]
else:
    cxx_flags += [
        "-O3", 
        "-std=c++17"
    ]    
    nvcc_flags += [
        "-O3",
        "-std=c++17"
    ]

# -------------------------------------------------
# CUDA / ROCm specific
# -------------------------------------------------
if IS_HIP:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    nvcc_flags += [f"--offload-arch={arch}" for arch in archs]
else:
    # CUDA only
    if IS_WINDOWS:
        nvcc_flags += ["-allow-unsupported-compiler"]

# -------------------------------------------------
# Extensions
# -------------------------------------------------
ext_modules = [

    # ===============================
    # Main CuMesh extension
    # ===============================
    CUDAExtension(
        name="cumesh._C",
        sources=[
            "src/hash/hash.cu",

            "src/atlas.cu",
            "src/clean_up.cu",
            "src/cumesh.cu",
            "src/connectivity.cu",
            "src/geometry.cu",
            "src/io.cu",
            "src/simplify.cu",
            "src/shared.cu",

            "src/remesh/simple_dual_contour.cu",
            "src/remesh/svox2vert.cu",

            "src/ext.cpp",
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),

    # ===============================
    # cubvh
    # ===============================
    CUDAExtension(
        name="cumesh._cubvh",
        sources=[
            "third_party/cubvh/src/bvh.cu",
            "third_party/cubvh/src/api_gpu.cu",
            "third_party/cubvh/src/bindings.cpp",
        ],
        include_dirs=[
            os.path.join(ROOT, "third_party/cubvh/include"),
            os.path.join(ROOT, "third_party/cubvh/third_party/eigen"),
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags + [
                # The following definitions must be undefined
                # since we need half-precision operation.
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    ),

    # ===============================
    # xatlas (CPU only)
    # ===============================
    CUDAExtension(
        name="cumesh._cumesh_xatlas",
        sources=[
            "third_party/xatlas/xatlas.cpp",
            "third_party/xatlas/binding.cpp",
        ],
        extra_compile_args={
            "cxx": cxx_flags,
        },
    ),
]

# -------------------------------------------------
# Setup
# -------------------------------------------------
setup(
    name="cumesh",
    packages=["cumesh"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
