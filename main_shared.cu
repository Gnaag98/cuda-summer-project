#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common.hpp"

/// Requirement: blockDim.x == cell_particle_count.
__global__ void add_density_shared(const FloatingPoint *pos_x, const FloatingPoint *pos_y,
        FloatingPoint *density) {
    const auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= N) {
        return;
    }

    const auto index = threadIdx.x;

    // XXX: Assumes one block per cell.
    const auto cell_origin = uint2{ blockIdx.x % U, blockIdx.x / U };
    // Node indices.
    const auto index_bottom_left = get_node_index(cell_origin.x, cell_origin.y);
    const auto index_bottom_right = get_node_index(cell_origin.x +1, cell_origin.y);
    const auto index_top_left = get_node_index(cell_origin.x, cell_origin.y + 1);
    const auto index_top_right = get_node_index(cell_origin.x + 1, cell_origin.y + 1);

    const auto x = pos_x[particle_index];
    const auto y = pos_y[particle_index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_relative_cell = FloatingPoint2{ u - cell_origin.x, v - cell_origin.y };
    const auto weight_bottom_left  = (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y);
    const auto weight_bottom_right =      pos_relative_cell.x  * (1 - pos_relative_cell.y);
    const auto weight_top_left     = (1 - pos_relative_cell.x) *      pos_relative_cell.y;
    const auto weight_top_right    =      pos_relative_cell.x  *      pos_relative_cell.y;

    // Each particle will contribute to 4 cells.
    __shared__ FloatingPoint density_shared[4][block_size];

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

    if (index == 0) {
        atomicAdd(&density[index_bottom_left], density_shared[0][0]);
        atomicAdd(&density[index_bottom_right], density_shared[1][0]);
        atomicAdd(&density[index_top_left], density_shared[2][0]);
        atomicAdd(&density[index_top_right], density_shared[3][0]);
    }
}

void store_density(std::filesystem::path filepath,
                   std::span<const FloatingPoint> density) {
    auto density_file = std::ofstream(filepath);
    for (int row = 0; row < (V + 1); ++row) {
        for (int col = 0; col < (U + 1); ++col) {
            density_file << density[row * (U + 1) + col] << ',';
        }
        density_file << '\n';
    }
}

void store_debug(std::filesystem::path filepath,
                   std::span<const FloatingPoint> shared) {
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
    auto h_pos_x = std::vector<FloatingPoint>(positions_count);
    auto h_pos_y = std::vector<FloatingPoint>(positions_count);
    auto h_density = std::vector<FloatingPoint>(lattice_count);

    // Allocate particle positions and densities on the device.
    FloatingPoint *d_pos_x;
    FloatingPoint *d_pos_y;
    FloatingPoint *d_density;
    cudaMalloc(&d_pos_x, positions_bytes);
    cudaMalloc(&d_pos_y, positions_bytes);
    cudaMalloc(&d_density, lattice_bytes);

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

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_density(output_directory / "density_shared.csv", h_density);
}
