// HIP device-mode shim for Eigen: provide std algorithms in global namespace
// when __HIP_DEVICE_COMPILE__ is set (Eigen uses ::fill_n, ::copy_n etc.)
#pragma once
#if defined(EIGEN_HIP_DEVICE_COMPILE) || defined(__HIP_DEVICE_COMPILE__)
#include <algorithm>
using std::fill_n;
using std::copy_n;
using std::copy;
using std::fill;
using std::min;
using std::max;
#endif
