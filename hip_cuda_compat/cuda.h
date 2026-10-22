// CUDA -> HIP shim for Windows ROCm builds (TheRock SDK lacks cuda compat headers)
#pragma once
#include <hip/hip_runtime.h>

// CUDA runtime type aliases
typedef hipStream_t cudaStream_t;
typedef hipError_t cudaError_t;
typedef hipEvent_t cudaEvent_t;
typedef hipDeviceProp_t cudaDeviceProp;
typedef hipMemcpyKind cudaMemcpyKind;

// CUDA error code aliases
static constexpr auto cudaSuccess = hipSuccess;
static constexpr auto cudaErrorNotReady = hipErrorNotReady;
static constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
static constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
static constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
static constexpr auto cudaMemcpyDefault = hipMemcpyDefault;

// Template cudaMalloc matches CUDA's typed-pointer overload used by cubvh gpu_memory.h
template <class T>
inline cudaError_t cudaMalloc(T** ptr, size_t size) {
    return hipMalloc(reinterpret_cast<void**>(ptr), size);
}
inline cudaError_t cudaFree(void* ptr) {
    return hipFree(ptr);
}
template <class T>
inline cudaError_t cudaMallocHost(T** ptr, size_t size) {
    return hipHostMalloc(reinterpret_cast<void**>(ptr), size, 0);
}
inline cudaError_t cudaFreeHost(void* ptr) {
    return hipHostFree(ptr);
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return hipMemcpy(dst, src, count, (hipMemcpyKind)kind);
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
    return hipMemcpyAsync(dst, src, count, (hipMemcpyKind)kind, stream);
}
inline cudaError_t cudaMemset(void* dst, int value, size_t count) {
    return hipMemset(dst, value, count);
}
inline cudaError_t cudaMemsetAsync(void* dst, int value, size_t count, cudaStream_t stream = 0) {
    return hipMemsetAsync(dst, value, count, stream);
}
inline cudaError_t cudaDeviceSynchronize() {
    return hipDeviceSynchronize();
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return hipStreamSynchronize(stream);
}
inline cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return hipStreamQuery(stream);
}
inline cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    return hipStreamGetPriority(stream, priority);
}
inline cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    return hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
}
inline cudaError_t cudaGetLastError() {
    return hipGetLastError();
}
inline cudaError_t cudaPeekAtLastError() {
    return hipPeekAtLastError();
}
inline const char* cudaGetErrorString(cudaError_t err) {
    return hipGetErrorString(err);
}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    return hipGetDeviceProperties(prop, device);
}
inline cudaError_t cudaSetDevice(int device) {
    return hipSetDevice(device);
}
inline cudaError_t cudaGetDevice(int* device) {
    return hipGetDevice(device);
}
inline cudaError_t cudaGetDeviceCount(int* count) {
    return hipGetDeviceCount(count);
}
