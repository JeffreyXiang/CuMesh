#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "api.h"
#include "../utils.h"
#include "../hash/hash.cuh"

// Maximum number of intersections per voxel (12 edges)
#define MAX_INTERSECTIONS 12
// Maximum number of normal clusters for sharp edge detection
#define MAX_CLUSTERS 4


// Helper: normalize a float3 vector
__device__ __forceinline__ float3 normalize_float3(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-8f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0.0f, 0.0f, 1.0f); // fallback
}


// Helper: dot product of two float3
__device__ __forceinline__ float dot_float3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


template<typename T>
__device__ __forceinline__ float get_vertex_val(
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const float* __restrict__ udf,
    const size_t N_vert,
    int x, int y, int z,
    int W, int H, int D
) {
    size_t flat_idx = (size_t)x * H * D + (size_t)y * D + z;
    T key = static_cast<T>(flat_idx);
    uint32_t idx = linear_probing_lookup(hashmap_keys, hashmap_vals, key, N_vert);
    return udf[idx];
}


template<typename T>
static __global__ void simple_dual_contour_kernel(
    const size_t N_vert,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const int32_t* __restrict__ coords,
    const float* __restrict__ udf,
    float* __restrict__ out_vertices,
    int32_t* __restrict__ out_intersected
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= M) return;

    int vx = coords[thread_id * 3 + 0];
    int vy = coords[thread_id * 3 + 1];
    int vz = coords[thread_id * 3 + 2];

    float3 intersection_sum = make_float3(0.0f, 0.0f, 0.0f);
    int intersection_count = 0;

    // Traverse the 12 edges of the voxel
    // Axis X
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx, vy + u, vz + v, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + 1, vy + u, vz + v, W, H, D);

            // Calculate the intersection point
            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                // P = P1 + t * (P2 - P1)
                intersection_sum.x += (float)vx + t;
                intersection_sum.y += (float)(vy + u);
                intersection_sum.z += (float)(vz + v);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 0] = 1;
                }
                else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 0] = -1;
                }
                else {
                    out_intersected[thread_id * 3 + 0] = 0;
                }
            }
        }
    }

    // Axis Y
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy, vz + v, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy + 1, vz + v, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += (float)(vx + u);
                intersection_sum.y += (float)vy + t;
                intersection_sum.z += (float)(vz + v);
                intersection_count++;
            }
            
            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 1] = 1;
                }
                else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 1] = -1;
                }
                else {
                    out_intersected[thread_id * 3 + 1] = 0;
                }
            }
        }
    }

    // Axis Z
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy + v, vz, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy + v, vz + 1, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += (float)(vx + u);
                intersection_sum.y += (float)(vy + v);
                intersection_sum.z += (float)vz + t;
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 2] = 1;
                }
                else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 2] = -1;
                }
                else {
                    out_intersected[thread_id * 3 + 2] = 0;
                }
            }
        }
    }

    // Calculate the mean intersection point
    if (intersection_count > 0) {
        out_vertices[thread_id * 3 + 0] = intersection_sum.x / intersection_count;
        out_vertices[thread_id * 3 + 1] = intersection_sum.y / intersection_count;
        out_vertices[thread_id * 3 + 2] = intersection_sum.z / intersection_count;
    } else {
        // Fallback: Voxel Center
        out_vertices[thread_id * 3 + 0] = (float)vx + 0.5f;
        out_vertices[thread_id * 3 + 1] = (float)vy + 0.5f;
        out_vertices[thread_id * 3 + 2] = (float)vz + 0.5f;
    }
}


/**
 * Sharp edge preserving dual contouring kernel.
 * Uses UDF gradients to detect sharp features and clusters intersections by normal.
 */
template<typename T>
static __global__ void simple_dual_contour_sharp_kernel(
    const size_t N_vert,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const int32_t* __restrict__ coords,
    const float* __restrict__ udf,
    const float cos_sharp_threshold,  // cos(sharp_angle_threshold)
    float* __restrict__ out_vertices,
    int32_t* __restrict__ out_intersected
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= M) return;

    int vx = coords[thread_id * 3 + 0];
    int vy = coords[thread_id * 3 + 1];
    int vz = coords[thread_id * 3 + 2];

    // Storage for intersection points and their gradients (normals)
    float3 intersections[MAX_INTERSECTIONS];
    float3 gradients[MAX_INTERSECTIONS];
    int intersection_count = 0;

    // Compute gradient using the edge direction and sign change
    // This is more robust than querying neighbors that may not exist in sparse hashmap
    #define COMPUTE_GRADIENT_FROM_EDGE(v1x, v1y, v1z, v2x, v2y, v2z, val1, val2, grad) do { \
        float edge_dx = (float)(v2x - v1x); \
        float edge_dy = (float)(v2y - v1y); \
        float edge_dz = (float)(v2z - v1z); \
        float edge_len = sqrtf(edge_dx*edge_dx + edge_dy*edge_dy + edge_dz*edge_dz); \
        if (edge_len > 1e-6f) { \
            float slope = (val2 - val1) / edge_len; \
            if (slope > 0) { \
                grad = make_float3(edge_dx/edge_len, edge_dy/edge_len, edge_dz/edge_len); \
            } else { \
                grad = make_float3(-edge_dx/edge_len, -edge_dy/edge_len, -edge_dz/edge_len); \
            } \
        } else { \
            grad = make_float3(0.0f, 0.0f, 1.0f); \
        } \
    } while(0)

    // Traverse the 12 edges of the voxel - Axis X
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            int v1x = vx, v1y = vy + u, v1z = vz + v;
            int v2x = vx + 1, v2y = vy + u, v2z = vz + v;
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, v1x, v1y, v1z, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, v2x, v2y, v2z, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                float3 pt = make_float3((float)vx + t, (float)(vy + u), (float)(vz + v));
                intersections[intersection_count] = pt;
                COMPUTE_GRADIENT_FROM_EDGE(v1x, v1y, v1z, v2x, v2y, v2z, val1, val2, gradients[intersection_count]);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 0] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 0] = -1;
                } else {
                    out_intersected[thread_id * 3 + 0] = 0;
                }
            }
        }
    }

    // Axis Y
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            int v1x = vx + u, v1y = vy, v1z = vz + v;
            int v2x = vx + u, v2y = vy + 1, v2z = vz + v;
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, v1x, v1y, v1z, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, v2x, v2y, v2z, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                float3 pt = make_float3((float)(vx + u), (float)vy + t, (float)(vz + v));
                intersections[intersection_count] = pt;
                COMPUTE_GRADIENT_FROM_EDGE(v1x, v1y, v1z, v2x, v2y, v2z, val1, val2, gradients[intersection_count]);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 1] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 1] = -1;
                } else {
                    out_intersected[thread_id * 3 + 1] = 0;
                }
            }
        }
    }

    // Axis Z
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            int v1x = vx + u, v1y = vy + v, v1z = vz;
            int v2x = vx + u, v2y = vy + v, v2z = vz + 1;
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, v1x, v1y, v1z, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, v2x, v2y, v2z, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                float3 pt = make_float3((float)(vx + u), (float)(vy + v), (float)vz + t);
                intersections[intersection_count] = pt;
                COMPUTE_GRADIENT_FROM_EDGE(v1x, v1y, v1z, v2x, v2y, v2z, val1, val2, gradients[intersection_count]);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 2] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 2] = -1;
                } else {
                    out_intersected[thread_id * 3 + 2] = 0;
                }
            }
        }
    }

    #undef COMPUTE_GRADIENT_FROM_EDGE

    if (intersection_count == 0) {
        // Fallback: Voxel Center
        out_vertices[thread_id * 3 + 0] = (float)vx + 0.5f;
        out_vertices[thread_id * 3 + 1] = (float)vy + 0.5f;
        out_vertices[thread_id * 3 + 2] = (float)vz + 0.5f;
        return;
    }

    // Cluster gradients by angular similarity (greedy clustering)
    int cluster_ids[MAX_INTERSECTIONS];
    float3 cluster_centers[MAX_CLUSTERS];
    float3 cluster_pos_sum[MAX_CLUSTERS];
    int cluster_counts[MAX_CLUSTERS];
    int num_clusters = 0;

    for (int i = 0; i < intersection_count; i++) {
        float3 g = gradients[i];
        bool found_cluster = false;

        for (int c = 0; c < num_clusters; c++) {
            float d = dot_float3(g, cluster_centers[c]);
            if (d >= cos_sharp_threshold) {
                // Add to existing cluster
                cluster_ids[i] = c;
                // Update cluster center (running average)
                float3 new_center = make_float3(
                    cluster_centers[c].x + g.x,
                    cluster_centers[c].y + g.y,
                    cluster_centers[c].z + g.z
                );
                cluster_centers[c] = normalize_float3(new_center);
                // Accumulate position
                cluster_pos_sum[c].x += intersections[i].x;
                cluster_pos_sum[c].y += intersections[i].y;
                cluster_pos_sum[c].z += intersections[i].z;
                cluster_counts[c]++;
                found_cluster = true;
                break;
            }
        }

        if (!found_cluster && num_clusters < MAX_CLUSTERS) {
            // Create new cluster
            cluster_ids[i] = num_clusters;
            cluster_centers[num_clusters] = g;
            cluster_pos_sum[num_clusters] = intersections[i];
            cluster_counts[num_clusters] = 1;
            num_clusters++;
        } else if (!found_cluster) {
            // Too many clusters, add to closest
            float best_dot = -1.0f;
            int best_c = 0;
            for (int c = 0; c < num_clusters; c++) {
                float d = dot_float3(g, cluster_centers[c]);
                if (d > best_dot) {
                    best_dot = d;
                    best_c = c;
                }
            }
            cluster_ids[i] = best_c;
            cluster_pos_sum[best_c].x += intersections[i].x;
            cluster_pos_sum[best_c].y += intersections[i].y;
            cluster_pos_sum[best_c].z += intersections[i].z;
            cluster_counts[best_c]++;
        }
    }

    // Compute final vertex position
    if (num_clusters == 1) {
        // Single cluster: simple average (original behavior)
        out_vertices[thread_id * 3 + 0] = cluster_pos_sum[0].x / cluster_counts[0];
        out_vertices[thread_id * 3 + 1] = cluster_pos_sum[0].y / cluster_counts[0];
        out_vertices[thread_id * 3 + 2] = cluster_pos_sum[0].z / cluster_counts[0];
    } else {
        // Multiple clusters: sharp feature detected
        // Solve simplified QEF: find point that minimizes distance to all planes
        // Each plane is defined by (cluster_avg_pos, cluster_center_normal)

        // Use iterative method: start from global centroid, project onto each plane
        float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
        float total_weight = 0.0f;

        for (int c = 0; c < num_clusters; c++) {
            float w = (float)cluster_counts[c];
            centroid.x += cluster_pos_sum[c].x;
            centroid.y += cluster_pos_sum[c].y;
            centroid.z += cluster_pos_sum[c].z;
            total_weight += w;
        }
        centroid.x /= total_weight;
        centroid.y /= total_weight;
        centroid.z /= total_weight;

        // Compute plane equations for each cluster
        float3 plane_points[MAX_CLUSTERS];
        for (int c = 0; c < num_clusters; c++) {
            plane_points[c] = make_float3(
                cluster_pos_sum[c].x / cluster_counts[c],
                cluster_pos_sum[c].y / cluster_counts[c],
                cluster_pos_sum[c].z / cluster_counts[c]
            );
        }

        // Iterative projection (3 iterations is usually enough)
        float3 vertex = centroid;
        for (int iter = 0; iter < 5; iter++) {
            float3 correction = make_float3(0.0f, 0.0f, 0.0f);

            for (int c = 0; c < num_clusters; c++) {
                float3 n = cluster_centers[c];
                float3 p = plane_points[c];

                // Distance from vertex to plane
                float dist = (vertex.x - p.x) * n.x +
                            (vertex.y - p.y) * n.y +
                            (vertex.z - p.z) * n.z;

                // Move vertex towards plane
                float w = (float)cluster_counts[c] / total_weight;
                correction.x -= dist * n.x * w;
                correction.y -= dist * n.y * w;
                correction.z -= dist * n.z * w;
            }

            vertex.x += correction.x;
            vertex.y += correction.y;
            vertex.z += correction.z;
        }

        // Clamp to voxel bounds (with small margin)
        vertex.x = fmaxf((float)vx - 0.1f, fminf((float)vx + 1.1f, vertex.x));
        vertex.y = fmaxf((float)vy - 0.1f, fminf((float)vy + 1.1f, vertex.y));
        vertex.z = fmaxf((float)vz - 0.1f, fminf((float)vz + 1.1f, vertex.z));

        out_vertices[thread_id * 3 + 0] = vertex.x;
        out_vertices[thread_id * 3 + 1] = vertex.y;
        out_vertices[thread_id * 3 + 2] = vertex.z;
    }
}


/**
 * Isosurfacing a volume defined on vertices of a sparse voxel grid using a simple dual contouring algorithm.
 * Dual vertices are computed by mean of edge intersections.
 * 
 * @param hashmap_keys  [Nvert] uint32/uint64 hashmap of the vertices keys
 * @param hashmap_vals  [Nvert] uint32 tensor containing the hashmap values as vertex indices
 * @param coords        [Mvox, 3] int32 tensor containing the coordinates of the active voxels
 * @param udf           [Mvert] float tensor containing the UDF/SDF values at the vertices
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 *
 * @return              [L, 3] float tensor containing the active vertices (Dual Vertices)
                        [L, 3] int32 tensor containing the intersected edges (1: intersected, 0: not intersected)
 */
std::tuple<torch::Tensor, torch::Tensor> cumesh::simple_dual_contour(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& udf,
    int W,
    int H,
    int D
) {
    const size_t M = coords.size(0);
    const size_t N_vert = hashmap_keys.size(0);

    auto vertices = torch::empty({(long)M, 3}, torch::dtype(torch::kFloat32).device(coords.device()));
    auto intersected = torch::empty({(long)M, 3}, torch::dtype(torch::kInt32).device(coords.device()));

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (hashmap_keys.dtype() == torch::kUInt32) {
        simple_dual_contour_kernel<<<blocks, threads>>>(
            N_vert,
            M,
            W, H, D,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            udf.data_ptr<float>(),
            vertices.data_ptr<float>(),
            intersected.data_ptr<int32_t>()
        );
    } 
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        simple_dual_contour_kernel<<<blocks, threads>>>(
            N_vert,
            M,
            W, H, D,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            udf.data_ptr<float>(),
            vertices.data_ptr<float>(),
            intersected.data_ptr<int32_t>()
        );
    } 
    else {
        TORCH_CHECK(false, "Unsupported hashmap data type");
    }

    CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(vertices, intersected);
}


/**
 * Isosurfacing with sharp edge preservation using gradient-based normal clustering.
 *
 * @param hashmap_keys  [Nvert] uint32/uint64 hashmap of the vertices keys
 * @param hashmap_vals  [Nvert] uint32 tensor containing the hashmap values as vertex indices
 * @param coords        [Mvox, 3] int32 tensor containing the coordinates of the active voxels
 * @param udf           [Mvert] float tensor containing the UDF/SDF values at the vertices
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param sharp_angle_threshold  angle in degrees above which edges are considered sharp
 *
 * @return              [L, 3] float tensor containing the active vertices (Dual Vertices)
 *                      [L, 3] int32 tensor containing the intersected edges (1: intersected, 0: not intersected)
 */
std::tuple<torch::Tensor, torch::Tensor> cumesh::simple_dual_contour_sharp(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& udf,
    int W,
    int H,
    int D,
    float sharp_angle_threshold
) {
    const size_t M = coords.size(0);
    const size_t N_vert = hashmap_keys.size(0);

    // Convert angle to cosine threshold (cos of angle in radians)
    // Edges with dot product < cos_threshold are considered sharp
    float cos_threshold = cosf(sharp_angle_threshold * 3.14159265358979f / 180.0f);

    auto vertices = torch::empty({(long)M, 3}, torch::dtype(torch::kFloat32).device(coords.device()));
    auto intersected = torch::empty({(long)M, 3}, torch::dtype(torch::kInt32).device(coords.device()));

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (hashmap_keys.dtype() == torch::kUInt32) {
        simple_dual_contour_sharp_kernel<<<blocks, threads>>>(
            N_vert,
            M,
            W, H, D,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            udf.data_ptr<float>(),
            cos_threshold,
            vertices.data_ptr<float>(),
            intersected.data_ptr<int32_t>()
        );
    }
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        simple_dual_contour_sharp_kernel<<<blocks, threads>>>(
            N_vert,
            M,
            W, H, D,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            udf.data_ptr<float>(),
            cos_threshold,
            vertices.data_ptr<float>(),
            intersected.data_ptr<int32_t>()
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported hashmap data type");
    }

    CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(vertices, intersected);
}
