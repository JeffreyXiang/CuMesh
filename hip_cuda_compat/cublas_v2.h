// CUDA -> HIP shim for Windows ROCm builds
#pragma once
#include <hipblas/hipblas.h>
typedef hipblasHandle_t cublasHandle_t;
typedef hipblasStatus_t cublasStatus_t;
