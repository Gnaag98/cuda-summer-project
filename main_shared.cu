#include <iostream>
#include <vector>

#include "common.cuh"

struct KernelData {
    std::vector<uint> first_instruction_index_per_block;
    std::vector<uint> first_cell_index_per_block;
    std::vector<uint> cell_count_per_block;
    std::vector<uint> first_particle_index_per_cell;
    std::vector<uint> particle_count_per_cell;
};

// TODO: Give this function a beter name.
// XXX: This implementation will result in blocks sharing cells often, since
// each block will handle block_size amount of threads each. A better solution
// would be to only allow sharing cells when particles per cell > block size,
// that is it is impossible for a block to single-handedly process the cell.
auto get_kernel_data(std::span<const uint> cell_indices) {
    auto first_instruction_index_per_block = std::vector<uint>();
    auto first_cell_index_per_block = std::vector<uint>();
    auto cell_count_per_block = std::vector<uint>();
    auto first_particle_index_per_cell = std::vector<uint>{};
    auto particle_count_per_cell = std::vector<uint>{};
    // TODO: Find a clever formula for the reserve sizes.
    first_particle_index_per_cell.reserve(cell_count);
    particle_count_per_cell.reserve(cell_count);
    first_cell_index_per_block.reserve(N / block_size);
    cell_count_per_block.reserve(N / block_size);
    first_instruction_index_per_block.reserve(N / block_size);

    auto current_cell_index = cell_indices[0];
    auto block_particle_count = 0;

    auto particle_offset = 0;
    auto particle_count = 0;
    auto instruction_offset = 0;
    auto cell_offset = 0;

    auto occupied_cells_count = 0;
    auto max_particles_per_cell = 0;
    auto max_cells_per_block = 0;

    /// Return true if all indices are fully processed.
    const auto is_all_done = [&](const int i){
        return i == cell_indices.size();
    };
    /// Return true if the block is full.
    const auto is_block_done = [&](const int i){
        return is_all_done(i) || block_particle_count == block_size;
    };
    /// Return true if the cell is fully processed.
    const auto is_cell_done = [&](const int i){
        // Makes sure to do the bounds check before the access.
        return is_all_done(i) || cell_indices[i] != current_cell_index;
    };

    auto i = 0;
    while (i < cell_indices.size()) {
        // Count particles in cell as long as possible.
        while (!is_block_done(i) && !is_cell_done(i)) {
            ++particle_count;
            ++block_particle_count;
            ++i;
        }
        const auto was_cell_done = is_cell_done(i);
        const auto was_block_done = is_block_done(i);

        // Count number of processed/semi-processed cells before the cell index
        // might change.
        const auto cell_count = current_cell_index - cell_offset + 1;
        // Add particle data for processed/semi-processed cell.
        first_particle_index_per_cell.emplace_back(particle_offset);
        particle_count_per_cell.emplace_back(particle_count);
        max_particles_per_cell = max(particle_count, max_particles_per_cell);
        particle_offset += particle_count;
        particle_count = 0;
        // Cell fully processed or nothing more to process.
        if (was_cell_done) {
            // Mark the next cell for processing.
            current_cell_index = cell_indices[i];
            ++occupied_cells_count;
        }
        // Block full or nothing more to process.
        if (was_block_done) {
            first_cell_index_per_block.emplace_back(cell_offset);
            cell_count_per_block.emplace_back(cell_count);
            max_cells_per_block = max(cell_count, max_cells_per_block);
            first_instruction_index_per_block.emplace_back(instruction_offset);
            instruction_offset += cell_count;
            // Only add fully processed cells to the offset. This way the next
            // block can continue processing the semi-processed cell.
            const auto fully_processed_count = cell_count - !was_cell_done;
            cell_offset += fully_processed_count;
            block_particle_count = 0;
        }
    }
    return KernelData{
        .first_instruction_index_per_block = std::move(first_instruction_index_per_block),
        .first_cell_index_per_block = std::move(first_cell_index_per_block),
        .cell_count_per_block = std::move(cell_count_per_block),
        .first_particle_index_per_cell = std::move(first_particle_index_per_cell),
        .particle_count_per_cell = std::move(particle_count_per_cell),
        // TODO: Add these back when/if relevant.
        /* .occupied_cells_count = occupied_cells_count,
        .max_particles_per_cell = max_particles_per_cell,
        .max_cells_per_block = max_cells_per_block */
    };
}

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

__device__
auto get_cell_index_rel_block(
        const uint first_instruction_in_block_index,
        const uint block_cell_count,
        const uint *particle_count_per_cell) {
    auto cell_first_particle_index = 0;
    for (auto i = 0u; i < block_cell_count; ++i) {
        const auto instruction_index = first_instruction_in_block_index + i;
        const auto cell_particle_count = particle_count_per_cell[instruction_index];
        if (threadIdx.x < cell_first_particle_index + cell_particle_count) {
            return i;
        }
        cell_first_particle_index += cell_particle_count;
    }
    // Return an invalid index otherwise.
    return block_cell_count;
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

__global__ void add_density_shared(
        const FloatingPoint *pos_x,
        const FloatingPoint *pos_y,
        FloatingPoint *density,
        const uint *first_instruction_index_per_block,
        const uint *first_cell_index_per_block,
        const uint *cell_count_per_block,
        const uint *first_particle_index_per_cell,
        const uint *particle_count_per_cell
    ) {
    const auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= N) return;


    // Each particle will contribute to 4 cells.
    __shared__ FloatingPoint density_shared[4][block_size];

    // Reset the shared memory, since inactive threads will not overrwite
    // garbage values.
    density_shared[0][threadIdx.x] = 0;
    density_shared[1][threadIdx.x] = 0;
    density_shared[2][threadIdx.x] = 0;
    density_shared[3][threadIdx.x] = 0;
    __syncthreads();

    // Block-specific variables.
    const auto first_instruction_in_block_index = first_instruction_index_per_block[blockIdx.x];
    const auto first_cell_in_block_index = first_cell_index_per_block[blockIdx.x];
    const auto block_cell_count = cell_count_per_block[blockIdx.x];

    // Cell-specific variables.
    // XXX: Assumes both fixed and equal number of cells per block.
    const auto cell_index_rel_block = get_cell_index_rel_block(
        first_instruction_in_block_index, block_cell_count,
        particle_count_per_cell);
    if (cell_index_rel_block >= block_cell_count) {
        printf("Error! Block %d: Could not assign cell index to thread %d.\n",
            blockIdx.x, threadIdx.x);
        return;
    }
    const auto instruction_index = first_instruction_in_block_index + cell_index_rel_block;
    const auto cell_index = first_cell_in_block_index + cell_index_rel_block;
    const auto first_particle_in_cell_index = first_particle_index_per_cell[instruction_index];
    const auto cell_particle_count = particle_count_per_cell[instruction_index];
    
    // Thread-specific variables.
    const auto particle_index_rel_cell = particle_index - first_particle_in_cell_index;
    
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
        // Make sure not to stride outside of the cell range. Crucial when
        // the number of particles in a cell isn't a power of two.
        if (particle_index_rel_cell < stride && particle_index_rel_cell + stride < cell_particle_count) {
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

    get_cell_index_per_particle<<<
        block_count, block_size
    >>>(
        d_pos_x, d_pos_y, d_cell_indices
    );
    load(h_cell_indices, d_cell_indices);
    auto kernel_data = get_kernel_data(h_cell_indices);

    /* std::cout << "first_instruction_index_per_block: ";
    for (const auto v : kernel_data.first_instruction_index_per_block) {
        std::cout << v << ", ";
    }
    std::cout << '\n';
    std::cout << "first_cell_index_per_block: ";
    for (const auto v : kernel_data.first_cell_index_per_block) {
        std::cout << v << ", ";
    }
    std::cout << '\n';
    std::cout << "cell_count_per_block: ";
    for (const auto v : kernel_data.cell_count_per_block) {
        std::cout << v << ", ";
    }
    std::cout << '\n';
    std::cout << "first_particle_index_per_cell: ";
    for (const auto v : kernel_data.first_particle_index_per_cell) {
        std::cout << v << ", ";
    }
    std::cout << '\n';
    std::cout << "particle_count_per_cell: ";
    for (const auto v : kernel_data.particle_count_per_cell) {
        std::cout << v << ", ";
    }
    std::cout << '\n';

    std::cout << kernel_data.first_instruction_index_per_block.size() << '\n';
    std::cout << kernel_data.first_cell_index_per_block.size() << '\n';
    std::cout << kernel_data.cell_count_per_block.size() << '\n';
    std::cout << kernel_data.first_particle_index_per_cell.size() << '\n';
    std::cout << kernel_data.particle_count_per_cell.size() << '\n'; */

    // XXX: This is a mess.first_instruction_index_per_block
    decltype(kernel_data.first_instruction_index_per_block)::value_type *d_first_instruction_index_per_block;
    decltype(kernel_data.first_cell_index_per_block)::value_type *d_first_cell_index_per_block;
    decltype(kernel_data.cell_count_per_block)::value_type *d_cell_count_per_block;
    decltype(kernel_data.first_particle_index_per_cell)::value_type *d_first_particle_index_per_cell;
    decltype(kernel_data.particle_count_per_cell)::value_type *d_particle_count_per_cell;
    allocate_array(&d_first_instruction_index_per_block, kernel_data.first_instruction_index_per_block.size());
    allocate_array(&d_first_cell_index_per_block, kernel_data.first_cell_index_per_block.size());
    allocate_array(&d_cell_count_per_block, kernel_data.cell_count_per_block.size());
    allocate_array(&d_first_particle_index_per_cell, kernel_data.first_particle_index_per_cell.size());
    allocate_array(&d_particle_count_per_cell, kernel_data.particle_count_per_cell.size());
    store(d_first_instruction_index_per_block, kernel_data.first_instruction_index_per_block);
    store(d_first_cell_index_per_block, kernel_data.first_cell_index_per_block);
    store(d_cell_count_per_block, kernel_data.cell_count_per_block);
    store(d_first_particle_index_per_cell, kernel_data.first_particle_index_per_cell);
    store(d_particle_count_per_cell, kernel_data.particle_count_per_cell);

    add_density_shared<<<
        block_count, block_size
    >>>(
        d_pos_x, d_pos_y, d_density,
        d_first_instruction_index_per_block,
        d_first_cell_index_per_block, d_cell_count_per_block,
        d_first_particle_index_per_cell, d_particle_count_per_cell
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
