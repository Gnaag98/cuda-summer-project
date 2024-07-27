#include <chrono>
#include <iostream>
#include <vector>

#include <cub/device/device_radix_sort.cuh>

#include "common.cuh"

template<typename T>
void debug_store_array(std::filesystem::path filepath,
        std::span<const T> data) {
    auto file = std::ofstream(filepath);
    for (auto i = 0; i < data.size(); ++i) {
        file << data[i] << ',';
    }
    file << ";\n";
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

__global__
void initialize_indices(uint *indices, const uint N) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    indices[index] = index;
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
    // Use min() to force particles back inside the grid of cell.
    const auto cell_origin = uint2{
        min(static_cast<uint>(floor(x_to_u(x))), U - 1),
        min(static_cast<uint>(floor(y_to_v(y))), V - 1)
    };
    cell_indices[particle_index] = cell_origin.x + cell_origin.y * U;
}

__global__
void initialize_kernel_data(
        const uint *cell_indices,
        const uint particle_count,
        uint *particle_indices_rel_cell,
        uint *particle_count_per_cell) {
    const auto indirect_particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (indirect_particle_index >= particle_count) return;

    __shared__ uint shared_cell_indices[block_size];

    __shared__ uint shared_particle_indices_rel_cell[block_size];
    // Incrementing 0, 1, 2, ..., for each new cell.
    __shared__ uint cell_ids[block_size];
    // Indexed with cell id, not cell index.
    __shared__ uint particle_count_per_cell_id[block_size];

    const auto global_cell_index = cell_indices[indirect_particle_index];
    shared_cell_indices[threadIdx.x] = global_cell_index;
    __syncthreads();

    /* if (blockIdx.x == 16410) {
        printf("----- %d\n", threadIdx.x);
    } */

    if (threadIdx.x == 0) {
        shared_particle_indices_rel_cell[0] = 0;
        cell_ids[0] = 0;
        // Start on 1 since we already set the value for i = 0.
        auto particle_index_rel_cell = 1;
        auto cell_id = 0;
        auto cell_particle_count = 0;
        auto previous_cell_index = shared_cell_indices[0];

        // TODO: Don't loop too far in last block.
        const auto block_particle_count = min(
            particle_count - indirect_particle_index, block_size);
        for (auto i = 1u; i < block_particle_count; ++i) {
            const auto cell_index = shared_cell_indices[i];
            ++cell_particle_count;
            if (cell_index > previous_cell_index) {
                particle_count_per_cell_id[cell_id] = cell_particle_count;
                particle_index_rel_cell = 0;
                ++cell_id;
                cell_particle_count = 0;
                previous_cell_index = cell_index;
            }
            shared_particle_indices_rel_cell[i] = particle_index_rel_cell++;
            cell_ids[i] = cell_id;
        }
        particle_count_per_cell_id[cell_id] = cell_particle_count + 1;
    }
    __syncthreads();
    particle_indices_rel_cell[indirect_particle_index]
        = shared_particle_indices_rel_cell[threadIdx.x];
    const auto cell_index = cell_ids[threadIdx.x];
    particle_count_per_cell[indirect_particle_index]
        = particle_count_per_cell_id[cell_index];
}

__global__
void add_density_shared(
        const FloatingPoint *pos_x,
        const FloatingPoint *pos_y,
        const uint particle_count,
        FloatingPoint *density,
        const uint *cell_indices,
        const uint *particle_indices,
        const uint *particle_indices_rel_cell,
        const uint *particle_count_per_cell
    ) {
    const auto indirect_particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (indirect_particle_index >= particle_count) return;

    // Each particle will contribute to 4 cells.
    __shared__ FloatingPoint density_shared[4][block_size];

    const auto cell_index = cell_indices[indirect_particle_index];
    const auto particle_index = particle_indices[indirect_particle_index];
    const auto particle_index_rel_cell = particle_indices_rel_cell[indirect_particle_index];
    const auto cell_particle_count = particle_count_per_cell[indirect_particle_index];
    
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
    auto h_cell_indices_before = std::vector<uint>(N);
    auto h_cell_indices_after = std::vector<uint>(N);
    auto h_density = std::vector<FloatingPoint>(node_count);

    // Allocate on the device.
    auto d_pos_x = (decltype(h_pos_x)::value_type *){};
    auto d_pos_y = (decltype(h_pos_y)::value_type *){};
    auto d_particle_indices_before = (uint *){};
    auto d_particle_indices_after = (uint *){};
    auto d_cell_indices_before = (decltype(h_cell_indices_before)::value_type *){};
    auto d_cell_indices_after = (decltype(h_cell_indices_before)::value_type *){};
    auto d_particle_indices_rel_cell = (uint *){};
    auto d_particle_count_per_cell = (uint *){};
    auto d_density = (decltype(h_density)::value_type *){};
    allocate_array(&d_pos_x, h_pos_x.size());
    allocate_array(&d_pos_y, h_pos_y.size());
    allocate_array(&d_particle_indices_before, h_pos_x.size());
    allocate_array(&d_particle_indices_after, h_pos_x.size());
    allocate_array(&d_cell_indices_before, h_cell_indices_before.size());
    allocate_array(&d_cell_indices_after, h_cell_indices_before.size());
    allocate_array(&d_particle_indices_rel_cell, h_pos_x.size());
    allocate_array(&d_particle_count_per_cell, h_pos_x.size());
    allocate_array(&d_density, h_density.size());

    const auto block_count = (N + block_size - 1) / block_size;
    printf("N: %d, block_count: %d, block_size: %d\n", N, block_count, block_size);

    // Determine temporary device storage requirements
    void *d_sort_storage = nullptr;
    auto sort_storage_byte_count = size_t{};
    cub::DeviceRadixSort::SortPairs(
        d_sort_storage, sort_storage_byte_count,
        d_cell_indices_before, d_cell_indices_after,
        d_particle_indices_before, d_particle_indices_after,
        N
    );
    // Allocate temporary storage
    cudaMalloc(&d_sort_storage, sort_storage_byte_count);

    // Distribute the cells using shuffled indices to force uncoalesced global
    // access when reading particle postions.
    const auto distribution_indices = get_shuffled_indices(h_pos_x.size());
    distribute_from_density(h_pos_x, h_pos_y, distribution_indices,
        particle_count_per_cell);

    initialize_indices<<<block_count, block_size>>>(
        d_particle_indices_before, N
    );

    // Copy from the host to the device.
    store(d_pos_x, h_pos_x);
    store(d_pos_y, h_pos_y);

    cudaDeviceSynchronize();
    // Perform multiple iterations and pretend the particles are moving as well.
    for (auto i = 0; i < iteration_count; ++i) {
        using namespace std::chrono;
        const auto start_time = high_resolution_clock::now();

        // Reset density.
        fill(d_density, 0, h_density.size());

        get_cell_index_per_particle<<<
            block_count, block_size
        >>>(
            d_pos_x, d_pos_y, N, d_cell_indices_before
        );

        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(
            d_sort_storage, sort_storage_byte_count,
            d_cell_indices_before, d_cell_indices_after,
            d_particle_indices_before, d_particle_indices_after,
            N
        );

        initialize_kernel_data<<<
            block_count, block_size
        >>>(
            d_cell_indices_after, N,
            d_particle_indices_rel_cell, d_particle_count_per_cell);

        auto h_particle_indices_rel_cell = std::vector<uint>{};
        auto h_particle_count_per_cell = std::vector<uint>{};
        h_particle_indices_rel_cell.reserve(N);
        h_particle_count_per_cell.reserve(N);
        load(h_particle_indices_rel_cell, d_particle_indices_rel_cell);
        load(h_particle_count_per_cell, d_particle_count_per_cell);

        add_density_shared<<<
            block_count, block_size
        >>>(
            d_pos_x, d_pos_y, N, d_density,
            d_cell_indices_after,
            d_particle_indices_after,
            d_particle_indices_rel_cell,
            d_particle_count_per_cell
        );
        load(h_density, d_density);

        const auto end_time = high_resolution_clock::now();
        const auto duration = end_time - start_time;
        const auto duration_ms = duration_cast<milliseconds>(duration).count();
        const auto duration_us = duration_cast<microseconds>(duration).count();
        if (duration_ms == 0) {
            printf("Iteration %d took %ld us.\n", i,
                duration_us);
        } else {
            printf("Iteration %d took %ld.%ld ms.\n", i,
                duration_ms,
                duration_us);
        }
    }

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_particle_indices_before);
    cudaFree(d_cell_indices_before);
    cudaFree(d_density);

    cudaFree(d_sort_storage);
    cudaFree(d_particle_indices_rel_cell);
    cudaFree(d_particle_count_per_cell);

#ifdef DEBUG_STORE_RESULTS
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_positions(output_directory / "positions_shared.csv", h_pos_x, h_pos_y);
    store_density(output_directory / "density_shared.csv", h_density);
#endif
}
