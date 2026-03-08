#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAStream.h>
#include "utils.h"
#include "cumesh.h"


namespace cumesh {

inline cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}


template<typename T>
__global__ void arange_kernel(T* array, int N, int stride=1) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    array[tid] = static_cast<T>(tid * stride);
}


template<typename T1, typename T2>
__global__ void cast_kernel(T1* input, T2* output, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = static_cast<T2>(input[tid]);
}


template<typename T>
__global__ void fill_kernel(T* array, int N, T value) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    array[tid] = value;
}


template<typename T>
__global__ void scatter_kernel(
    const int* indices,
    const T* values,
    const size_t N,
    T* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[indices[tid]] = values[tid];
}


template<typename T>
__global__ void index_kernel(
    const T* values,
    const int* indices,
    const size_t N,
    T* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = values[indices[tid]];
}


template<typename T>
__global__ void diff_kernel(
    const T* values,
    const size_t N,
    T* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = values[tid+1] - values[tid];
}


template<typename T>
__global__ void set_flag_kernel(
    const int* indices,
    const size_t N,
    T* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[indices[tid]] = static_cast<T>(1);
}


template<typename CompT, typename FlagT, typename Comparator>
__global__ void compare_kernel(
    const CompT* values,
    const CompT threshold,
    const size_t N,
    Comparator op,
    FlagT* flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    flag[tid] = op(values[tid], threshold) ? static_cast<FlagT>(1) : static_cast<FlagT>(0);
}


template<typename Ta, typename Tb>
__global__ void inplace_div_kernel(
    Ta* a,
    const Tb* b,
    const size_t N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    a[tid] = a[tid] / static_cast<float>(b[tid]);
}


/**
 * Hook edges
 * @param adj: the buffer for adjacency, shape (M)
 * @param M: the number of adjacency
 * @param conn_comp_ids: the buffer for connected component ids, shape (F)
 * @param end_flag: flag to indicate if any union operation happened
 */
__global__ void hook_edges_kernel(
    const int2* adj,
    const int M,
    int* conn_comp_ids,
    int* end_flag
);


/**
 * Compress connected components
 * @param conn_comp_ids: the buffer for connected component ids, shape (F)
 * @param F: the number of faces
 */
__global__ void compress_components_kernel(
    int* conn_comp_ids,
    const int F
);


template<typename T>
__global__ void get_diff_kernel(
    const T* ids_sorted,
    T* ids_diff,
    const int N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (tid == N-1) {
        ids_diff[tid] = 1;
        return;
    }
    if (ids_sorted[tid] != ids_sorted[tid+1]) {
        ids_diff[tid] = 1;
    } else {
        ids_diff[tid] = 0;
    }
}


template<typename T>
int compress_ids(T* ids, size_t N, Buffer<char>& cub_temp_storage, T* inverse=nullptr) {
    cudaStream_t stream = current_stream();
    int *cu_indices, *cu_indices_argsorted;
    int *cu_num = nullptr;
    T *cu_ids_sorted;
    CUDA_CHECK(cudaMalloc(&cu_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_indices_argsorted, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_ids_sorted, N * sizeof(T)));
    arange_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(cu_indices, N);
    CUDA_CHECK(cudaGetLastError());
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        ids, cu_ids_sorted,
        cu_indices, cu_indices_argsorted,
        N, 0, sizeof(T) * 8, stream
    ));
    cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        cub_temp_storage.ptr, temp_storage_bytes,
        ids, cu_ids_sorted,
        cu_indices, cu_indices_argsorted,
        N, 0, sizeof(T) * 8, stream
    ));
    // get diff
    T* cu_new_ids;
    CUDA_CHECK(cudaMalloc(&cu_new_ids, N * sizeof(T)));
    get_diff_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        cu_ids_sorted,
        cu_new_ids,
        N
    );
    CUDA_CHECK(cudaGetLastError());

    // inverse
    if (inverse) {
        CUDA_CHECK(cudaMalloc(&cu_num, sizeof(int)));
        temp_storage_bytes = 0;
        CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, temp_storage_bytes,
            cu_ids_sorted, cu_new_ids, inverse, cu_num,
            N, stream
        ));
        cub_temp_storage.resize(temp_storage_bytes);
        CUDA_CHECK(cub::DeviceSelect::Flagged(
            cub_temp_storage.ptr, temp_storage_bytes,
            cu_ids_sorted, cu_new_ids, inverse, cu_num,
            N, stream
        ));
    }

    // scan diff
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_new_ids,
        N, stream
    ));
    cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_ids,
        N, stream
    ));

    // scatter
    scatter_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        cu_indices_argsorted,
        cu_new_ids,
        N,
        ids
    );
    CUDA_CHECK(cudaGetLastError());
    T num_components;
    CUDA_CHECK(cudaMemcpyAsync(&num_components, cu_new_ids + N-1, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    num_components += 1;

    // Free all scratch memory — stream is synced, all GPU work is done
    CUDA_CHECK(cudaFree(cu_indices));
    CUDA_CHECK(cudaFree(cu_ids_sorted));
    CUDA_CHECK(cudaFree(cu_new_ids));
    CUDA_CHECK(cudaFree(cu_indices_argsorted));
    if (cu_num) CUDA_CHECK(cudaFree(cu_num));

    return static_cast<int>(num_components);
}


// DEBUG

template <typename T>
void print_max_val(T* ptr, size_t size) {
    cudaStream_t stream = current_stream();
    T* dbg_cu_max_val;
    CUDA_CHECK(cudaMalloc(&dbg_cu_max_val, sizeof(T)));
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Max(
        nullptr, temp_storage_bytes,
        ptr,
        dbg_cu_max_val,
        size, stream
    ));
    char* temp_storage;
    CUDA_CHECK(cudaMalloc(&temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceReduce::Max(
        temp_storage, temp_storage_bytes,
        ptr,
        dbg_cu_max_val,
        size, stream
    ));
    T h_max_val;
    CUDA_CHECK(cudaMemcpyAsync(&h_max_val, dbg_cu_max_val, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "Max value: " << h_max_val << std::endl;
    CUDA_CHECK(cudaFree(dbg_cu_max_val));
    CUDA_CHECK(cudaFree(temp_storage));
}

template <typename T>
void print_val(T* ptr, size_t size) {
    cudaStream_t stream = current_stream();
    T h_ptr[size];
    CUDA_CHECK(cudaMemcpyAsync(h_ptr, ptr, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < size; i++) {
        std::cout << h_ptr[i] << " ";
    }
    std::cout << std::endl;
}


} // namespace cumesh
