#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common.hpp"

/// Requirement: blockDim.x == cell_particle_count.
__global__ void add_density_shared(const float *pos_x, const float *pos_y,
        float *density) {
    const auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= N) {
        return;
    }

    const auto index = threadIdx.x;

    const auto x = pos_x[particle_index];
    const auto y = pos_y[particle_index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);


    // Node coordinates.
    const auto node_bottom_left  = int2{ static_cast<int>(floor(u)), static_cast<int>(floor(v)) };
    const auto node_bottom_right = int2{ static_cast<int>( ceil(u)), static_cast<int>(floor(v)) };
    const auto node_top_left     = int2{ static_cast<int>(floor(u)), static_cast<int>( ceil(v)) };
    const auto node_top_right    = int2{ static_cast<int>( ceil(u)), static_cast<int>( ceil(v)) };

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    // NOTE: Error-prone calculation of cell origin. If point on border this
    // could result in the origin of a neighboring cell.
    //const auto cell_origin = float2{ node_bottom_left.x, node_bottom_left.y };

    // Better calculation of cell origin based on block id.
    // XXX: Assumes one block per cell.
    const auto cell_origin = uint2{ blockIdx.x % U, blockIdx.x / U };
    
    const auto pos_relative_cell = float2{ u - cell_origin.x, v - cell_origin.y };
    const auto weight_bottom_left  = (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y);
    const auto weight_bottom_right =      pos_relative_cell.x  * (1 - pos_relative_cell.y);
    const auto weight_top_left     = (1 - pos_relative_cell.x) *      pos_relative_cell.y;
    const auto weight_top_right    =      pos_relative_cell.x  *      pos_relative_cell.y;

    // Node indices.
    const auto index_bottom_left = get_node_index(node_bottom_left.x, node_bottom_left.y);
    const auto index_bottom_right = get_node_index(node_bottom_right.x, node_bottom_right.y);
    const auto index_top_left = get_node_index(node_top_left.x, node_top_left.y);
    const auto index_top_right = get_node_index(node_top_right.x, node_top_right.y);

    // Each particle will contribute to 4 cells.
    __shared__ float density_shared[4][block_size];

    density_shared[0][index] = weight_bottom_left;
    density_shared[1][index] = weight_bottom_right;
    density_shared[2][index] = weight_top_left;
    density_shared[3][index] = weight_top_right;
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (index < stride) {
            density_shared[0][index] += density_shared[0][index + stride];
            density_shared[1][index] += density_shared[1][index + stride];
            density_shared[2][index] += density_shared[2][index + stride];
            density_shared[3][index] += density_shared[3][index + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&density[index_bottom_left], density_shared[0][0]);
        atomicAdd(&density[index_bottom_right], density_shared[1][0]);
        atomicAdd(&density[index_top_left], density_shared[2][0]);
        atomicAdd(&density[index_top_right], density_shared[3][0]);
    }
}

__global__ void add_density_debug(const float *pos_x, const float *pos_y,
        float *density, float *density_shared) {
    const auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= N) {
        return;
    }

    const auto x = pos_x[particle_index];
    const auto y = pos_y[particle_index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);

    // Node coordinates.
    const auto node_bottom_left  = int2{ static_cast<int>(floor(u)), static_cast<int>(floor(v)) };
    const auto node_bottom_right = int2{ static_cast<int>( ceil(u)), static_cast<int>(floor(v)) };
    const auto node_top_left     = int2{ static_cast<int>(floor(u)), static_cast<int>( ceil(v)) };
    const auto node_top_right    = int2{ static_cast<int>( ceil(u)), static_cast<int>( ceil(v)) };

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_relative_cell = float2{ u - node_bottom_left.x, v - node_bottom_left.y };
    const auto weight_bottom_left  = (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y);
    const auto weight_bottom_right =      pos_relative_cell.x  * (1 - pos_relative_cell.y);
    const auto weight_top_left     = (1 - pos_relative_cell.x) *      pos_relative_cell.y;
    const auto weight_top_right    =      pos_relative_cell.x  *      pos_relative_cell.y;

    // Node indices.
    const auto index_bottom_left = get_node_index(node_bottom_left.x, node_bottom_left.y);
    const auto index_bottom_right = get_node_index(node_bottom_right.x, node_bottom_right.y);
    const auto index_top_left = get_node_index(node_top_left.x, node_top_left.y);
    const auto index_top_right = get_node_index(node_top_right.x, node_top_right.y);

    density_shared[particle_index + 0 * N] = weight_bottom_left;
    density_shared[particle_index + 1 * N] = weight_bottom_right;
    density_shared[particle_index + 2 * N] = weight_top_left;
    density_shared[particle_index + 3 * N] = weight_top_right;
    /* __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (index < stride) {
            density_shared[particle_index + 0 * N] += density_shared[particle_index + stride + 0 * N];
            density_shared[particle_index + 1 * N] += density_shared[particle_index + stride + 1 * N];
            density_shared[particle_index + 2 * N] += density_shared[particle_index + stride + 2 * N];
            density_shared[particle_index + 3 * N] += density_shared[particle_index + stride + 3 * N];
        }
        __syncthreads();
    }

    if (true) {
        atomicAdd(&density[index_bottom_left], 0);
        atomicAdd(&density[index_bottom_right], 0);
        atomicAdd(&density[index_top_left], 0);
        atomicAdd(&density[index_top_right], 0);
    } */
}

void store_density(std::filesystem::path filepath,
                   std::span<const float> density) {
    auto density_file = std::ofstream(filepath);
    for (int row = 0; row < (V + 1); ++row) {
        for (int col = 0; col < (U + 1); ++col) {
            density_file << density[row * (U + 1) + col] << ',';
        }
        density_file << '\n';
    }
}

void store_debug(std::filesystem::path filepath,
                   std::span<const float> shared) {
    auto density_file = std::ofstream(filepath);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < N; ++col) {
            density_file << shared[row * N + col] << ',';
        }
        density_file << '\n';
    }
}

int main() {
    // Allocate particle positions and densities on the host.
    auto h_pos_x = std::vector<float>(positions_count);
    auto h_pos_y = std::vector<float>(positions_count);
    auto h_density = std::vector<float>(lattice_count);

    // Allocate particle positions and densities on the device.
    float *d_pos_x;
    float *d_pos_y;
    float *d_density;
    cudaMalloc(&d_pos_x, positions_bytes);
    cudaMalloc(&d_pos_y, positions_bytes);
    cudaMalloc(&d_density, lattice_bytes);

    auto h_shared = std::vector<float>(4 * N);
    float *d_shared;
    cudaMalloc(&d_shared, 4 * N * sizeof(float));

    distribute_random(h_pos_x, h_pos_y);

    // Copy positions from the host to the device.
    cudaMemcpy(d_pos_x, h_pos_x.data(), positions_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y.data(), positions_bytes, cudaMemcpyHostToDevice);

    // Initialize density.
    cudaMemset(d_density, 0, lattice_bytes);

    add_density_shared<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_density);
    //add_density_debug<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_density, d_shared);
    //cudaDeviceSynchronize();
    cudaMemcpy(h_density.data(), d_density, lattice_bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_shared.data(), d_shared, 4 * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);

    cudaFree(d_shared);

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_density(output_directory / "density_shared.csv", h_density);
    store_debug(output_directory / "debug_shared.csv", h_shared);
}
