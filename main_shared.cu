#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common.hpp"

/// XXX: Requirement: particles per cell > blockDim.x.
__global__ void add_density_shared(const FloatingPoint *pos_x, const FloatingPoint *pos_y,
        FloatingPoint *density) {
    // Each particle will contribute to 4 cells.
    __shared__ FloatingPoint density_shared[4][block_size];

    // Reset the shared memory, since inactive threads will not overrwite
    // garbage values.
    density_shared[0][threadIdx.x] = 0;
    density_shared[1][threadIdx.x] = 0;
    density_shared[2][threadIdx.x] = 0;
    density_shared[3][threadIdx.x] = 0;
    __syncthreads();

    const auto cell_index = blockIdx.x / blocks_per_cell;
    const auto cell_block_index = blockIdx.x % blocks_per_cell;
    const auto cell_particle_index = cell_block_index * block_size + threadIdx.x;
    if (cell_particle_index >= cell_particle_count) {
        return;
    }
    const auto particle_index = cell_index * cell_particle_count + cell_block_index * blockDim.x + threadIdx.x;
    // XXX: This if statement might be unneccessary due to the above early
    // return, since the number of particles per cell is constant.
    if (particle_index >= N) {
        printf("This shouldn't happen! blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
        return;
    }

    const auto cell_origin = uint2{ cell_index % U, cell_index / U };
    // Node indices.
    const auto index_bottom_left  = get_node_index(cell_origin.x,     cell_origin.y);
    const auto index_bottom_right = get_node_index(cell_origin.x + 1, cell_origin.y);
    const auto index_top_left     = get_node_index(cell_origin.x,     cell_origin.y + 1);
    const auto index_top_right    = get_node_index(cell_origin.x + 1, cell_origin.y + 1);

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

    density_shared[0][threadIdx.x] = weight_bottom_left;
    density_shared[1][threadIdx.x] = weight_bottom_right;
    density_shared[2][threadIdx.x] = weight_top_left;
    density_shared[3][threadIdx.x] = weight_top_right;
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            density_shared[0][threadIdx.x] += density_shared[0][threadIdx.x + stride];
            density_shared[1][threadIdx.x] += density_shared[1][threadIdx.x + stride];
            density_shared[2][threadIdx.x] += density_shared[2][threadIdx.x + stride];
            density_shared[3][threadIdx.x] += density_shared[3][threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&density[index_bottom_left],  density_shared[0][0]);
        atomicAdd(&density[index_bottom_right], density_shared[1][0]);
        atomicAdd(&density[index_top_left],     density_shared[2][0]);
        atomicAdd(&density[index_top_right],    density_shared[3][0]);
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
    cudaMemcpy(h_density.data(), d_density, lattice_bytes, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_density(output_directory / "density_shared.csv", h_density);
}
