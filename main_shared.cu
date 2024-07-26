#include <iostream>
#include <vector>

#include <cub/device/device_radix_sort.cuh>

#include "common.cuh"

struct KernelData {
    std::vector<uint> first_instruction_index_per_block;
    std::vector<uint> cell_count_per_block;

    std::vector<uint> cell_index_per_instruction;
    std::vector<uint> first_particle_index_per_cell;
    std::vector<uint> particle_count_per_cell;
};

// TODO: Give this function a beter name.
// XXX: This implementation will result in blocks sharing cells often, since
// each block will handle block_size amount of threads each. A better solution
// would be to only allow sharing cells when particles per cell > block size,
// that is it is impossible for a block to single-handedly process the cell.
auto get_kernel_data(std::span<const uint> cell_indices) {
    if (cell_indices.size() == 0) {
        throw std::runtime_error("cell_indices empty.");
    }
    auto kernel_data = KernelData{};
    // TODO: Find a clever formula for the reserve sizes.
    kernel_data.first_particle_index_per_cell.reserve(cell_count);
    kernel_data.particle_count_per_cell.reserve(cell_count);

    kernel_data.cell_index_per_instruction.reserve(cell_indices.size() / block_size);
    kernel_data.cell_count_per_block.reserve(cell_indices.size() / block_size);
    kernel_data.first_instruction_index_per_block.reserve(cell_indices.size() / block_size);

    auto current_cell_index = cell_indices[0];
    auto block_particle_count = 0;

    auto particle_offset = 0;
    auto particle_count = 0;
    auto instruction_offset = 0;
    auto cell_offset = 0;
    auto cell_count = 0;


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
        // Don't start the block on this cell index if nothing was found.
        if (block_particle_count == 0) {
            ++cell_offset;
        }
        // Add particle data for processed/semi-processed cell.
        if (particle_count > 0) {
            ++cell_count;
            kernel_data.cell_index_per_instruction.emplace_back(current_cell_index);
            kernel_data.first_particle_index_per_cell.emplace_back(particle_offset);
            kernel_data.particle_count_per_cell.emplace_back(particle_count);
            particle_offset += particle_count;
            particle_count = 0;
        }
        // Block full or nothing more to process.
        if (was_block_done) {
            kernel_data.cell_count_per_block.emplace_back(cell_count);
            kernel_data.first_instruction_index_per_block.emplace_back(instruction_offset);
            instruction_offset += cell_count;
            // Only add fully processed cells to the offset. This way the next
            // block can continue processing the semi-processed cell.
            cell_offset = current_cell_index;
            block_particle_count = 0;
            cell_count = 0;
        }
        // Cell fully processed or nothing more to process.
        if (was_cell_done) {
            // Mark the next cell for processing.
            current_cell_index = cell_indices[i];
        }
    }
    return kernel_data;
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
        const FloatingPoint *pos_y, const uint particle_count,
        uint *cell_indices) {
    const auto particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
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
        const uint particle_count,
        FloatingPoint *density,
        const uint *particle_indices,
        const uint *first_instruction_index_per_block,
        const uint *cell_count_per_block,
        const uint *cell_index_per_instruction,
        const uint *first_particle_index_per_cell,
        const uint *particle_count_per_cell
    ) {
    const auto indirect_particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (indirect_particle_index >= particle_count) return;

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
    const auto block_cell_count = cell_count_per_block[blockIdx.x];

    // Cell-specific variables.
    // XXX: Assumes both fixed and equal number of cells per block.
    const auto cell_index_rel_block = get_cell_index_rel_block(
        first_instruction_in_block_index, block_cell_count,
        particle_count_per_cell);
    if (cell_index_rel_block >= block_cell_count) {
        printf("Error! Block %d, thread: %d, indirect particle index: %d: Could not assign cell index to thread.\n",
            blockIdx.x, threadIdx.x, indirect_particle_index);
        return;
    }
    const auto instruction_index = first_instruction_in_block_index + cell_index_rel_block;
    const auto cell_index = cell_index_per_instruction[instruction_index];
    const auto first_particle_in_cell_index = first_particle_index_per_cell[instruction_index];
    const auto cell_particle_count = particle_count_per_cell[instruction_index];
    
    // Thread-specific variables.
    const auto particle_index_rel_cell = indirect_particle_index - first_particle_in_cell_index;
    const auto particle_index = particle_indices[indirect_particle_index];
    
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
#ifndef DEBUG_DISTRIBUTION
    // Generate a particle density.
    auto particle_count_per_cell = std::vector<uint>(cell_count);
    const auto N = generate_particle_density(particle_count_per_cell);
#endif
    // Allocate on the host.
    auto h_pos_x = std::vector<FloatingPoint>(N);
    auto h_pos_y = std::vector<FloatingPoint>(N);
    const auto h_particle_indices_before = get_ordered_indices(N);
    auto h_cell_indices_before = std::vector<uint>(N);
    auto h_cell_indices_after = std::vector<uint>(N);
    auto h_density = std::vector<FloatingPoint>(node_count);

    // Allocate on the device.
    auto d_pos_x = (decltype(h_pos_x)::value_type *){};
    auto d_pos_y = (decltype(h_pos_y)::value_type *){};
    auto d_particle_indices_before = (decltype(h_particle_indices_before)::value_type *){};
    auto d_particle_indices_after = (decltype(h_particle_indices_before)::value_type *){};
    auto d_cell_indices_before = (decltype(h_cell_indices_before)::value_type *){};
    auto d_cell_indices_after = (decltype(h_cell_indices_before)::value_type *){};
    auto d_density = (decltype(h_density)::value_type *){};
    allocate_array(&d_pos_x, h_pos_x.size());
    allocate_array(&d_pos_y, h_pos_y.size());
    allocate_array(&d_particle_indices_before, h_particle_indices_before.size());
    allocate_array(&d_particle_indices_after, h_particle_indices_before.size());
    allocate_array(&d_cell_indices_before, h_cell_indices_before.size());
    allocate_array(&d_cell_indices_after, h_cell_indices_before.size());
    allocate_array(&d_density, h_density.size());

    // Distribute the cells using shuffled indices to force uncoalesced global
    // access when reading particle postions.
    const auto distribution_indices = get_shuffled_indices(h_pos_x.size());
    distribute_from_density(h_pos_x, h_pos_y, distribution_indices,
        particle_count_per_cell);

    // Copy from the host to the device.
    store(d_pos_x, h_pos_x);
    store(d_pos_y, h_pos_y);
    store(d_particle_indices_before, h_particle_indices_before);

    // Initialize density.
    fill(d_density, 0, h_density.size());

    const auto block_count = (N + block_size - 1) / block_size;
    printf("N: %d, block_count: %d, block_size: %d\n", N, block_count, block_size);

    get_cell_index_per_particle<<<
        block_count, block_size
    >>>(
        d_pos_x, d_pos_y, N, d_cell_indices_before
    );

    // Determine temporary device storage requirements
    void *d_temp_sort_storage = nullptr;
    auto temp_sort_storage_byte_count = size_t{};
    cub::DeviceRadixSort::SortPairs(
        d_temp_sort_storage, temp_sort_storage_byte_count,
        d_cell_indices_before, d_cell_indices_after,
        d_particle_indices_before, d_particle_indices_after,
        N
    );

    // Allocate temporary storage
    cudaMalloc(&d_temp_sort_storage, temp_sort_storage_byte_count);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(
        d_temp_sort_storage, temp_sort_storage_byte_count,
        d_cell_indices_before, d_cell_indices_after,
        d_particle_indices_before, d_particle_indices_after,
        N
    );

    load(h_cell_indices_after, d_cell_indices_after);
    auto kernel_data = get_kernel_data(h_cell_indices_after);

    // XXX: This is a mess.first_instruction_index_per_block
    decltype(kernel_data.first_instruction_index_per_block)::value_type *d_first_instruction_index_per_block;
    decltype(kernel_data.cell_count_per_block)::value_type *d_cell_count_per_block;

    decltype(kernel_data.cell_index_per_instruction)::value_type *d_cell_index_per_instruction;
    decltype(kernel_data.first_particle_index_per_cell)::value_type *d_first_particle_index_per_cell;
    decltype(kernel_data.particle_count_per_cell)::value_type *d_particle_count_per_cell;

    allocate_array(&d_first_instruction_index_per_block, kernel_data.first_instruction_index_per_block.size());
    allocate_array(&d_cell_count_per_block, kernel_data.cell_count_per_block.size());
    
    allocate_array(&d_cell_index_per_instruction, kernel_data.cell_index_per_instruction.size());
    allocate_array(&d_first_particle_index_per_cell, kernel_data.first_particle_index_per_cell.size());
    allocate_array(&d_particle_count_per_cell, kernel_data.particle_count_per_cell.size());

    store(d_first_instruction_index_per_block, kernel_data.first_instruction_index_per_block);
    store(d_cell_count_per_block, kernel_data.cell_count_per_block);
    
    store(d_cell_index_per_instruction, kernel_data.cell_index_per_instruction);
    store(d_first_particle_index_per_cell, kernel_data.first_particle_index_per_cell);
    store(d_particle_count_per_cell, kernel_data.particle_count_per_cell);

    printf("Cell count: %d, instruction count: %lu\n",
        cell_count, kernel_data.cell_index_per_instruction.size());

    add_density_shared<<<
        block_count, block_size
    >>>(
        d_pos_x, d_pos_y, N, d_density, d_particle_indices_after,
        d_first_instruction_index_per_block, d_cell_count_per_block,
        d_cell_index_per_instruction, d_first_particle_index_per_cell, d_particle_count_per_cell
    );
    load(h_density, d_density);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_particle_indices_before);
    cudaFree(d_cell_indices_before);
    cudaFree(d_density);

    cudaFree(d_temp_sort_storage);

    cudaFree(d_first_instruction_index_per_block);
    cudaFree(d_cell_count_per_block);
    cudaFree(d_cell_index_per_instruction);
    cudaFree(d_first_particle_index_per_cell);
    cudaFree(d_particle_count_per_cell);

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_positions(output_directory / "positions.csv", h_pos_x, h_pos_y);
    store_density(output_directory / "density_shared.csv", h_density);
}
