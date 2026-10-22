// Windows HIP build: compile xatlas via hipcc (clang) to avoid MSVC
// rejecting or mishandling inherited dllimport constructors from PyTorch headers.
#include "xatlas.cpp"
