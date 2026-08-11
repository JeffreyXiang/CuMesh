// Windows HIP build: compile xatlas binding via hipcc (clang) to avoid MSVC
// rejecting or mishandling inherited dllimport constructors from PyTorch headers.
#include "binding.cpp"
