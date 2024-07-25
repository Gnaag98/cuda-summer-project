#include <iostream>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "common.cuh"

/// Calculate the cell index of each particle.
__global__
void get_cell_index_per_particle(const FloatingPoint *pos_x,
        const FloatingPoint *pos_y, uint *cell_indices) {
    const auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= N) {
        return;
    }
    const auto x = pos_x[particle_index];
    const auto y = pos_y[particle_index];
    const auto cell_origin = uint2{
        static_cast<uint>(floor(x_to_u(x))),
        static_cast<uint>(floor(y_to_v(y)))
    };
    cell_indices[particle_index] = cell_origin.x + cell_origin.y * U;
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

int main() {
    // Allocate particle positions and densities on the host.
    auto h_pos_x = std::vector<FloatingPoint>(N);
    auto h_pos_y = std::vector<FloatingPoint>(N);
    auto h_cell_indices = std::vector<FloatingPoint>(N);
    auto h_density = std::vector<FloatingPoint>(node_count);

    // Allocate particle positions and densities on the device.
    FloatingPoint *d_pos_x;
    FloatingPoint *d_pos_y;
    uint *d_cell_indices;
    FloatingPoint *d_density;
    allocate_array(&d_pos_x, h_pos_x.size());
    allocate_array(&d_pos_y, h_pos_y.size());
    allocate_array(&d_cell_indices, h_cell_indices.size());
    allocate_array(&d_density, h_density.size());

    distribute_random(h_pos_x, h_pos_y);

    // Copy positions from the host to the device.
    store(d_pos_x, h_pos_x);
    store(d_pos_y, h_pos_y);

    // Initialize density.
    fill(d_density, 0, h_density.size());

    const auto index_kernel_block_count = (N + block_size - 1) / block_size;
    get_cell_index_per_particle<<<
        index_kernel_block_count, block_size
    >>>(
        d_pos_x, d_pos_y, d_cell_indices
    );
    
    const auto density_kernel_block_count = cell_count * blocks_per_cell;
    add_density_shared<<<
        density_kernel_block_count, block_size
    >>>(
        d_pos_x, d_pos_y, d_density
    );
    load(h_density, d_density);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_cell_indices);
    cudaFree(d_density);

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_positions(output_directory / "positions.csv", h_pos_x, h_pos_y);
    store_density(output_directory / "density_shared.csv", h_density);
}
