#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "common.hpp"

/// Calculate the cell index of each particle.
__global__
void get_cell_index_per_particle(const FloatingPoint *pos_x,
        const FloatingPoint *pos_y, size_t *cell_indices) {
    const auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= N) {
        return;
    }
    const auto x = pos_x[particle_index];
    const auto y = pos_y[particle_index];
    const auto cell_x = floor(x_to_u(x));
    const auto cell_y = floor(y_to_v(y));
    cell_indices[particle_index] = cell_x + cell_y * U;
}

__global__ void add_density_shared(const FloatingPoint *pos_x, const FloatingPoint *pos_y,
        FloatingPoint *density) {
    using namespace cooperative_groups;

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
    const int incides[] = {
        get_node_index(cell_origin.x,     cell_origin.y),
        get_node_index(cell_origin.x + 1, cell_origin.y),
        get_node_index(cell_origin.x,     cell_origin.y + 1),
        get_node_index(cell_origin.x + 1, cell_origin.y + 1)
    };

    const auto x = pos_x[particle_index];
    const auto y = pos_y[particle_index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_relative_cell = FloatingPoint2{ u - cell_origin.x, v - cell_origin.y };
    const FloatingPoint weights[] = {
        (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y),
             pos_relative_cell.x  * (1 - pos_relative_cell.y),
        (1 - pos_relative_cell.x) *      pos_relative_cell.y,
             pos_relative_cell.x  *      pos_relative_cell.y
    };

    using BlockReduce = cub::BlockReduce<FloatingPoint, block_size>;
    using TempStorage = BlockReduce::TempStorage;
    __shared__ TempStorage temp_storage;
    auto block_reduce = BlockReduce{ temp_storage };
    for (auto i = 0; i < 4; ++i) {
        const auto density_reduced = block_reduce.Sum(weights[i]);
        if (threadIdx.x == 0) {
            atomicAdd(&density[incides[i]],  density_reduced);
        }
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

int main() {
    // Allocate particle positions and densities on the host.
    auto h_pos_x = std::vector<FloatingPoint>(positions_count);
    auto h_pos_y = std::vector<FloatingPoint>(positions_count);
    auto h_cell_indices = std::vector<size_t>(positions_count);
    auto h_density = std::vector<FloatingPoint>(lattice_count);

    // Allocate particle positions and densities on the device.
    FloatingPoint *d_pos_x;
    FloatingPoint *d_pos_y;
    size_t *d_cell_indices;
    FloatingPoint *d_density;
    cudaMalloc(&d_pos_x, positions_bytes);
    cudaMalloc(&d_pos_y, positions_bytes);
    cudaMalloc(&d_cell_indices, positions_bytes);
    cudaMalloc(&d_density, lattice_bytes);

    distribute_random(h_pos_x, h_pos_y);

    // Copy positions from the host to the device.
    cudaMemcpy(d_pos_x, h_pos_x.data(), positions_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y.data(), positions_bytes, cudaMemcpyHostToDevice);

    // Initialize density.
    cudaMemset(d_density, 0, lattice_bytes);

    //get_cell_index_per_particle<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_cell_indices);

    add_density_shared<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_density);
    cudaMemcpy(h_density.data(), d_density, lattice_bytes, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_cell_indices);
    /* cudaFree(d_density); */

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_density(output_directory / "density_shared.csv", h_density);
}
