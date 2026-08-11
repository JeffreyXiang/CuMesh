// Windows HIP build: compile this binding via hipcc (clang) to avoid MSVC
// rejecting HIP __attribute__ extensions in torch/extension.h.
#include "ext.cpp"
