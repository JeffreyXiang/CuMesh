// cuSolverDn -> hipSolver shim for Windows ROCm builds
#pragma once
#include <hipsolver/hipsolver.h>

typedef hipsolverHandle_t cusolverDnHandle_t;
typedef hipsolverStatus_t cusolverStatus_t;

static constexpr auto CUSOLVER_STATUS_SUCCESS = HIPSOLVER_STATUS_SUCCESS;
