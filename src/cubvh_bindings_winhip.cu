// Windows HIP build: compile the cubvh pybind binding via hipcc (clang) to
// avoid MSVC rejecting HIP __attribute__ extensions in torch/extension.h.
// eigen_hip_compat.h (CuMesh, hip_cuda_compat/) provides std:: algorithms in
// the global namespace for the HIP device passes (Eigen SparseCore needs
// ::fill_n etc.). Lives in CuMesh's tree so the cubvh submodule stays pristine
// upstream; it pulls the binding from the vendored submodule by relative path.
#include <eigen_hip_compat.h>
#include "../third_party/cubvh/src/bindings.cpp"
