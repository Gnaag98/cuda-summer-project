#include <iostream>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "common.cuh"

/// https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
constexpr
auto next_pow2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    v += v == 0;
    return v;
}

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
    // Each particle will contribute to 4 cells.
    __shared__ FloatingPoint density_shared[4][block_size];

    // Reset the shared memory, since inactive threads will not overrwite
    // garbage values.
    density_shared[0][threadIdx.x] = 0;
    density_shared[1][threadIdx.x] = 0;
    density_shared[2][threadIdx.x] = 0;
    density_shared[3][threadIdx.x] = 0;
    __syncthreads();

    // XXX: Assumes both fixed and equal number of cells per block.
    const auto cells_per_block = cell_count / gridDim.x;
    
    // Block-specific variables.
    const auto first_cell_in_block_index = blockIdx.x * cells_per_block;
    const auto first_particle_in_block_index = blockIdx.x * blockDim.x;

    // Cell-specific variables.
    // XXX: Assumes both fixed and equal number of cells per block.
    const auto cell_index_rel_block = threadIdx.x / cell_particle_count;
    
    // Thread-specific variables.
    const auto particle_index_rel_block = threadIdx.x;
    const auto particle_index = first_particle_in_block_index + particle_index_rel_block;
    if (particle_index >= N) return;
    const auto cell_index = first_cell_in_block_index + cell_index_rel_block;
    const auto particle_index_rel_cell = particle_index_rel_block % cell_particle_count;

    /* printf(
        "block: %d, thread: %d | cell_index: %d = %d + %d\n",
        blockIdx.x, threadIdx.x, cell_index, first_cell_in_block_index, cell_index_rel_block
    ); */
    /* printf(
        "block: %d, thread: %d | first_cell_in_block_index: %d = %d + %d\n",
        blockIdx.x, threadIdx.x,
        first_cell_in_block_index, blockIdx.x, cells_per_block
    ); */
    /* printf(
        "block: %d, thread: %d | particle_index: %d, particle_index_rel_cell: %d, particle_index_rel_block: %d\n",
        blockIdx.x, threadIdx.x,
        particle_index, particle_index_rel_cell, particle_index_rel_block
    ); */
    
    const auto cell_origin = uint2{ cell_index % U, cell_index / U };
    // Node indices.
    const auto indices = uint4{
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

    density_shared[0][threadIdx.x] = weights[0];
    density_shared[1][threadIdx.x] = weights[1];
    density_shared[2][threadIdx.x] = weights[2];
    density_shared[3][threadIdx.x] = weights[3];
    __syncthreads();

    // in-place reduction in shared memory
    // XXX: Assumes both fixed and equal number of particles per cell.
    for (int stride = next_pow2(cell_particle_count) / 2; stride > 0; stride /= 2) {
        if (particle_index_rel_cell < stride) {
            density_shared[0][threadIdx.x] += density_shared[0][threadIdx.x + stride];
            density_shared[1][threadIdx.x] += density_shared[1][threadIdx.x + stride];
            density_shared[2][threadIdx.x] += density_shared[2][threadIdx.x + stride];
            density_shared[3][threadIdx.x] += density_shared[3][threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (particle_index_rel_cell == 0) {
        atomicAdd(&density[indices.x], density_shared[0][threadIdx.x]);
        atomicAdd(&density[indices.y], density_shared[1][threadIdx.x]);
        atomicAdd(&density[indices.z], density_shared[2][threadIdx.x]);
        atomicAdd(&density[indices.w], density_shared[3][threadIdx.x]);
    }
}

int main() {
    // Allocate particle positions and densities on the host.
    auto h_pos_x = std::vector<FloatingPoint>(N);
    auto h_pos_y = std::vector<FloatingPoint>(N);
    auto h_cell_indices = std::vector<uint>(N);
    auto h_density = std::vector<FloatingPoint>(node_count);

    // Allocate particle positions and densities on the device.
    decltype(h_pos_x)::value_type *d_pos_x;
    decltype(h_pos_y)::value_type *d_pos_y;
    decltype(h_cell_indices)::value_type *d_cell_indices;
    decltype(h_density)::value_type *d_density;
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
    
    // XXX: Only valid if N % block_size == 0;
    const auto density_kernel_block_count = N / block_size;
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
