// Wrapper for ROCm hipsolver.h that also provides cuSolverDn type aliases
// needed by PyTorch ATen headers on Windows ROCm builds.
#pragma once
#include_next <hipsolver/hipsolver.h>

// ATen/cuda/CUDAContextLight.h references cusolverDnHandle_t when USE_ROCM is set;
// define it as the hip equivalent so the header compiles.
#ifndef cusolverDnHandle_t
typedef hipsolverDnHandle_t cusolverDnHandle_t;
#endif
#ifndef cusolverStatus_t
typedef hipsolverStatus_t cusolverStatus_t;
#endif
