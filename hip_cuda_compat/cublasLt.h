// CUDA -> HIP shim for Windows ROCm builds
#pragma once
#include <hipblaslt/hipblaslt.h>
typedef hipblasLtHandle_t cublasLtHandle_t;
