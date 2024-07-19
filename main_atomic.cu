#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common.hpp"

__global__
void add_density_atomic(const FloatingPoint *pos_x, const FloatingPoint *pos_y,
        FloatingPoint *density) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return;
    }

    const auto x = pos_x[index];
    const auto y = pos_y[index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);

    // Node coordinates.
    const auto node_bottom_left  = int2{ static_cast<int>(floor(u)), static_cast<int>(floor(v)) };
    const auto node_bottom_right = int2{ static_cast<int>( ceil(u)), static_cast<int>(floor(v)) };
    const auto node_top_left     = int2{ static_cast<int>(floor(u)), static_cast<int>( ceil(v)) };
    const auto node_top_right    = int2{ static_cast<int>( ceil(u)), static_cast<int>( ceil(v)) };

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_relative_cell = FloatingPoint2{ u - node_bottom_left.x, v - node_bottom_left.y };
    const auto weight_bottom_left  = (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y);
    const auto weight_bottom_right =      pos_relative_cell.x  * (1 - pos_relative_cell.y);
    const auto weight_top_left     = (1 - pos_relative_cell.x) *      pos_relative_cell.y;
    const auto weight_top_right    =      pos_relative_cell.x  *      pos_relative_cell.y;

    // Node indices.
    const auto index_bottom_left = get_node_index(node_bottom_left.x, node_bottom_left.y);
    const auto index_bottom_right = get_node_index(node_bottom_right.x, node_bottom_right.y);
    const auto index_top_left = get_node_index(node_top_left.x, node_top_left.y);
    const auto index_top_right = get_node_index(node_top_right.x, node_top_right.y);

    atomicAdd(&density[index_bottom_left], weight_bottom_left);
    atomicAdd(&density[index_bottom_right], weight_bottom_right);
    atomicAdd(&density[index_top_left], weight_top_left);
    atomicAdd(&density[index_top_right], weight_top_right);
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
