#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common.hpp"

__global__
void add_density_atomic(const FloatingPoint *pos_x, const FloatingPoint *pos_y,
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
    const auto weights = FloatingPoint4{
        (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y),
             pos_relative_cell.x  * (1 - pos_relative_cell.y),
        (1 - pos_relative_cell.x) *      pos_relative_cell.y,
             pos_relative_cell.x  *      pos_relative_cell.y
    };

    // Reduce densities per warp.
    auto tile = tiled_partition<warp_size>(this_thread_block());
    const auto densities_tile = tile_reduce<warp_size>(tile, weights);
    // Only the first thread of the tile holds the fully reduced sum.
    if (tile.thread_rank() == 0) {
        atomicAdd(&density[index_bottom_left], densities_tile.x);
        atomicAdd(&density[index_bottom_right], densities_tile.y);
        atomicAdd(&density[index_top_left], densities_tile.z);
        atomicAdd(&density[index_top_right], densities_tile.w);
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

    add_density_atomic<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_density);
    cudaMemcpy(h_density.data(), d_density, lattice_bytes, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_density(output_directory / "density_atomic.csv", h_density);
}
