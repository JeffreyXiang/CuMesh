// CUDA -> HIP shim for Windows ROCm builds
#pragma once
#include <hipsparse/hipsparse.h>
typedef hipsparseHandle_t cusparseHandle_t;
typedef hipsparseStatus_t cusparseStatus_t;
