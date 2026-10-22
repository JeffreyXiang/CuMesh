from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import platform
import sys


class _BuildExt(BuildExtension):
    """Register .hip with MSVC so hipified .cu->.hip sources compile on Windows.

    PyTorch's BuildExtension adds only .cu/.cuh to MSVC's _cpp_extensions, so a
    .hip source is rejected before hipcc runs (pytorch/pytorch#187665, not yet
    merged). No-op on Linux (clang, not MSVC). Forward-compatible with that PR
    (worst case a harmless duplicate .hip entry used only for membership tests).
    """
    def build_extensions(self):
        if sys.platform == "win32" and hasattr(self.compiler, "_cpp_extensions") \
                and ".hip" not in self.compiler._cpp_extensions:
            self.compiler._cpp_extensions.append(".hip")
        super().build_extensions()


ROOT = os.path.dirname(os.path.abspath(__file__))


def _hipify_cubvh_sources(sources, include_dirs):
    """Hipify the bundled cubvh .cu sources and return the .hip paths (Windows).

    torch's CUDAExtension already hipifies the extension's .cu sources, but its
    hipify pass prunes the entire third_party/ subtree from its file walk
    (torch/utils/hipify/hipify_python.py: matched_files_iter drops "third_party"),
    so files under third_party/cubvh/ are reported "not to be hipified" and the
    raw .cu is left for the compiler -- which fails because the raw .cu calls
    at::cuda::getCurrentCUDAStream() (only the hipified output uses
    at::hip::getCurrentHIPStreamMasqueradingAsCUDA()).

    We run hipify here with project_directory at the cubvh root, so the walk sees
    those files (no third_party prune at that root) and emits api_gpu.hip /
    bvh.hip with the correct stream rewrite. The scope has to cover cubvh's
    include/ tree as well as its src/ tree: cubvh's own gpu/api_gpu.h includes
    <ATen/cuda/CUDAContext.h>, and unless hipify rewrites that header too, the
    translation unit pulls in both ATen/cuda and ATen/hip, which declare the same
    c10::cuda::CUDAStream and cannot coexist. Hipifying the header yields
    gpu/api_gpu_hip.h with <ATen/hip/HIPContext.h> instead, which is exactly what
    the Linux build already compiles. The .hip / _hip.h outputs are generated
    build artifacts; the cubvh submodule source stays pristine and
    upstream-mergeable, and no manual stream shim is needed. _BuildExt registers
    .hip with MSVC so the outputs then compile. Linux is unaffected (torch
    hipifies cubvh in place there via its normal clang path; this helper is
    Windows-only).
    """
    from torch.utils.hipify import hipify_python

    cubvh_cu = [s for s in sources if s.endswith(".cu") and "cubvh" in s.replace("\\", "/")]
    if not cubvh_cu:
        return sources
    cubvh_root = os.path.join(ROOT, "third_party", "cubvh")
    result = hipify_python.hipify(
        project_directory=cubvh_root,
        output_directory=cubvh_root,
        header_include_dirs=include_dirs,
        includes=[
            os.path.join(cubvh_root, "src", "*"),
            os.path.join(cubvh_root, "include", "*"),
        ],
        # hipify appends extra_files entries to its file list verbatim, but
        # normalizes a candidate to forward slashes before the exact-match
        # test, so a backslash path from os.path.abspath() never matches and
        # the file is skipped. Only entries outside the walk root above depend
        # on this; the rest have a forward-slash duplicate from the walk.
        extra_files=[os.path.abspath(s).replace(os.sep, "/") for s in cubvh_cu],
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    rewritten = []
    for s in sources:
        s_abs = os.path.abspath(s)
        entry = result.get(s_abs)
        if entry is not None and entry.hipified_path is not None:
            rewritten.append(os.path.relpath(entry.hipified_path, ROOT).replace("\\", "/"))
        else:
            rewritten.append(s)
    return rewritten


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

if IS_WINDOWS and not IS_HIP:
    # Required for MSVC + nvcc + torch headers (CUDA build on Windows only)
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
    # HIP (Windows or Linux) and CUDA on Linux: clang-compatible flags
    cxx_flags += [
        "-O3",
        "-std=c++20"
    ]
    nvcc_flags += [
        "-O3",
        "-std=c++20"
    ]

# -------------------------------------------------
# CUDA / ROCm specific
# -------------------------------------------------
if IS_HIP:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    nvcc_flags += [f"--offload-arch={arch}" for arch in archs]
    if IS_WINDOWS:
        # TheRock SDK lacks cuda_cmake_macros.h (cmake-generated); skip it.
        nvcc_flags += ["-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE"]
else:
    # CUDA only
    if IS_WINDOWS:
        nvcc_flags += ["-allow-unsupported-compiler"]

# -------------------------------------------------
# Extra include dirs
# -------------------------------------------------
extra_include_dirs = []
if IS_WINDOWS and IS_HIP:
    # The TheRock Windows SDK lacks cuda compat headers (cuda_runtime.h,
    # cusparse.h, cublas_v2.h, etc.) that ROCm Linux provides under
    # /opt/rocm/include. Provide local shims that forward to hip/* equivalents.
    extra_include_dirs.append(os.path.join(ROOT, "hip_cuda_compat"))

# -------------------------------------------------
# cubvh extension sources
# -------------------------------------------------
cubvh_include_dirs = [
    os.path.join(ROOT, "third_party/cubvh/include"),
    os.path.join(ROOT, "third_party/cubvh/third_party/eigen"),
] + extra_include_dirs

cubvh_sources = [
    "third_party/cubvh/src/bvh.cu",
    "third_party/cubvh/src/api_gpu.cu",
    # On Windows+HIP, route the pybind binding through hipcc (clang) via a
    # CuMesh-owned wrapper (keeps the cubvh submodule pristine upstream).
    "src/cubvh_bindings_winhip.cu" if (IS_WINDOWS and IS_HIP) else "third_party/cubvh/src/bindings.cpp",
]

if IS_WINDOWS and IS_HIP:
    # torch's CUDAExtension hipify prunes third_party/, so it leaves cubvh's .cu
    # raw; hipify them here (see _hipify_cubvh_sources) and feed the .hip outputs.
    cubvh_sources = _hipify_cubvh_sources(cubvh_sources, cubvh_include_dirs)

# -------------------------------------------------
# cubvh symbol visibility
# -------------------------------------------------
# GCC/clang spelling; MSVC has no equivalent (nothing is exported unless it is
# dllexport-ed). nvcc parses the flags given to it as its own command line and
# rejects host-compiler flags, so the CUDA device pass has to forward them with
# -Xcompiler; hipcc is clang driven and takes them directly.
if IS_WINDOWS and not IS_HIP:
    visibility_flags = []
    visibility_nvcc_flags = []
else:
    visibility_flags = ["-fvisibility=hidden", "-fvisibility-inlines-hidden"]
    visibility_nvcc_flags = (
        visibility_flags if IS_HIP
        else [f"-Xcompiler={flag}" for flag in visibility_flags]
    )

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

            # On Windows+HIP, route the pybind binding through hipcc (clang) so
            # HIP __attribute__ extensions in torch/extension.h are accepted.
            "src/ext_winhip.cu" if (IS_WINDOWS and IS_HIP) else "src/ext.cpp",
        ],
        include_dirs=extra_include_dirs,
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
        sources=cubvh_sources,
        include_dirs=cubvh_include_dirs,
        extra_compile_args={
            # Hide cubvh's C++ symbols so this module's copy cannot interpose with
            # another cubvh in the same process; lets us consume the upstream-clean
            # cubvh (cubvh:: namespace) without a downstream namespace-wrap patch.
            "cxx": cxx_flags + visibility_flags,
            # NVCC-only flags (--extended-lambda, -U__CUDA_NO_HALF_*) are not
            # needed for hipcc; torch's hipify handles the half-precision types.
            "nvcc": (nvcc_flags if IS_HIP else nvcc_flags + [
                "--extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ]) + visibility_nvcc_flags,
        },
    ),

    # ===============================
    # xatlas (CPU only)
    # ===============================
    CUDAExtension(
        name="cumesh._cumesh_xatlas",
        sources=(
            # On Windows+HIP, compile via hipcc (clang) to avoid MSVC mishandling
            # of PyTorch headers with inherited dllimport constructors.
            [
                "third_party/xatlas/xatlas_winhip.cu",
                "third_party/xatlas/binding_winhip.cu",
            ]
            if (IS_WINDOWS and IS_HIP)
            else [
                "third_party/xatlas/xatlas.cpp",
                "third_party/xatlas/binding.cpp",
            ]
        ),
        extra_compile_args={
            "cxx": cxx_flags,
            # On Windows+HIP, the .cu sources go through hipcc; no GPU code in xatlas
            # so pass minimal flags (no --offload-arch needed).
            "nvcc": ["-O3", "-std=c++20", "-DC10_HIP_NO_CMAKE_CONFIGURE_FILE"] if (IS_WINDOWS and IS_HIP) else [],
        },
        include_dirs=extra_include_dirs,
    ),
]

# -------------------------------------------------
# Setup
# -------------------------------------------------
setup(
    name="cumesh",
    packages=["cumesh"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": _BuildExt},
)
