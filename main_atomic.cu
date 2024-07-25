#include <iostream>
#include <vector>

#include "common.cuh"

__global__
void add_density_atomic(const FloatingPoint *pos_x, const FloatingPoint *pos_y,
        const uint particle_count, FloatingPoint *density) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
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

    // Node indices.
    const auto index_bottom_left = get_node_index(node_bottom_left.x, node_bottom_left.y);
    const auto index_bottom_right = get_node_index(node_bottom_right.x, node_bottom_right.y);
    const auto index_top_left = get_node_index(node_top_left.x, node_top_left.y);
    const auto index_top_right = get_node_index(node_top_right.x, node_top_right.y);

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_relative_cell = FloatingPoint2{ u - node_bottom_left.x, v - node_bottom_left.y };
    const auto weights = FloatingPoint4{
        (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y),
             pos_relative_cell.x  * (1 - pos_relative_cell.y),
        (1 - pos_relative_cell.x) *      pos_relative_cell.y,
             pos_relative_cell.x  *      pos_relative_cell.y
    };

    atomicAdd(&density[index_bottom_left], weights.x);
    atomicAdd(&density[index_bottom_right], weights.y);
    atomicAdd(&density[index_top_left], weights.z);
    atomicAdd(&density[index_top_right], weights.w);
}

int main() {
#ifndef DEBUG_DISTRIBUTION
    // Generate a particle density.
    auto particle_count_per_cell = std::vector<uint>(cell_count);
    const auto N = generate_particle_density(particle_count_per_cell);
#endif
    // Allocate particle positions and densities on the host.
    auto h_pos_x = std::vector<FloatingPoint>(N);
    auto h_pos_y = std::vector<FloatingPoint>(N);
    auto h_density = std::vector<FloatingPoint>(node_count);

    // Allocate particle positions and densities on the device.
    decltype(h_pos_x)::value_type *d_pos_x;
    decltype(h_pos_y)::value_type *d_pos_y;
    decltype(h_density)::value_type *d_density;
    allocate_array(&d_pos_x, h_pos_x.size());
    allocate_array(&d_pos_y, h_pos_y.size());
    allocate_array(&d_density, h_density.size());

    distribute_from_density(h_pos_x, h_pos_y, particle_count_per_cell);

    // Copy positions from the host to the device.
    store(d_pos_x, h_pos_x);
    store(d_pos_y, h_pos_y);

    // Initialize density.
    fill(d_density, 0, h_density.size());

    const auto block_count = (N + block_size - 1) / block_size;
    printf("N: %d, block_count: %d, block_size: %d\n", N, block_count, block_size);
    add_density_atomic<<<block_count, block_size>>>(d_pos_x, d_pos_y, N, d_density);
    load(h_density, d_density);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_density);

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_density(output_directory / "density_atomic.csv", h_density);
}
