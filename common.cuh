#ifndef COMMON_HPP
#define COMMON_HPP

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include <cooperative_groups.h>

//#define DEBUG_DISTRIBUTION
//#define DEBUG_AVOID_EDGES
//#define DEBUG_STORE_RESULTS

// Select float or double for all floating point types.
using FloatingPoint = float;

using FloatingPoint2 = std::conditional<
    std::is_same<FloatingPoint, float>::value, float2, double2
>::type;
using FloatingPoint3 = std::conditional<
    std::is_same<FloatingPoint, float>::value, float3, double3
>::type;
using FloatingPoint4 = std::conditional<
    std::is_same<FloatingPoint, float>::value, float4, double4
>::type;

template<typename T>
struct Dimension {
    T width;
    T height;
};

template<typename T>
struct Rectangle {
    // (x, y) is the bottom left corner.
    T x;
    T y;
    T width;
    T height;

    constexpr static auto centered(const T width, const T height) {
        return Rectangle{
            .x = -width / 2,
            .y = -height / 2,
            .width = width,
            .height = height
        };
    }

    constexpr T left() const { return x; }
    constexpr T right() const { return x + width; }
    constexpr T bottom() const { return y; }
    constexpr T top() const { return y + height; }
};

const auto iteration_count = 1;

const auto mean_cell_particle_count = 16;
#ifndef DEBUG_DISTRIBUTION
// Number of cells in the grid.
const auto U = static_cast<int>(1024);
const auto V = static_cast<int>(4096);
const auto block_size = 128;
#else
const auto U = static_cast<int>(2);
const auto V = static_cast<int>(2);
const auto block_size = 4;
const uint particle_count_per_cell[] = { 3, 0, 2, 1 };
const auto N = ([]{
    auto n = 0;
    auto i = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            n += particle_count_per_cell[i];
            ++i;
        }
    }
    return n;
})();
#endif

// Size of 2D space.
constexpr auto space = Rectangle<FloatingPoint>::centered(3 * U, 5 * V);

const auto cell_count = U * V;

const auto cell = Dimension{
    .width = space.width / U,
    .height = space.height / V
};

// Density lattice, where each node is a cell corner.
const auto lattice = Dimension{ (U + 1), (V + 1) };
const auto node_count = lattice.width * lattice.height;

const auto random_seed = 1u;

const auto warp_size = 32;

template<typename T>
void allocate_array(T **device_pointer, size_t count) {
    cudaMalloc(device_pointer, count * sizeof(T));
}

template<typename T>
void fill(T *destination, const int value, const size_t count) {
    cudaMemset(destination, value, count * sizeof(T));
}

template<typename T>
void store(T *destination, const std::vector<T> &source) {
    cudaMemcpy(
        destination,
        source.data(),
        source.size() * sizeof(T),
        cudaMemcpyHostToDevice
    );
}

template<typename T>
void load(std::vector<T> &destination, const T *source) {
    cudaMemcpy(
        destination.data(),
        source,
        destination.size() * sizeof(T),
        cudaMemcpyDeviceToHost
    );
}

///Convert world x coordinate to horizontal cell index.
constexpr auto x_to_u(FloatingPoint x) {
    return (x - space.left()) * U / space.width;
}

///Convert world y coordinate to vertical cell index.
constexpr auto y_to_v(FloatingPoint y) {
    return (y + space.height / 2) * V / space.height;
}

constexpr auto get_node_index(const uint x, const uint y) {
    return x + y * (U + 1);
}

/// Return all indices in the range [0, indices_count - 1) in order.
auto get_ordered_indices(const uint indices_count) {
    auto indices = std::vector<uint>(indices_count);
    std::iota(indices.begin(), indices.end(), 0u);
    return indices;
}

/// Return all indices in the range [0, indices_count - 1) in random order.
auto get_shuffled_indices(const uint indices_count) {
    auto indices = get_ordered_indices(indices_count);
    auto random_engine = std::default_random_engine(random_seed);
    std::shuffle(indices.begin(), indices.end(), random_engine);
    return indices;
}

/// Generate a density distribution and return the total particle count.
auto generate_particle_density(std::span<uint> particle_count_per_cell) {
    auto random_engine = std::default_random_engine(random_seed);
    auto distribution = std::uniform_int_distribution<int>(
        0, 2 * mean_cell_particle_count
    );
    auto N = 0u;
    for (auto i = 0; i < particle_count_per_cell.size(); ++i) {
        const auto cell_particle_count = distribution(random_engine);
        particle_count_per_cell[i] = cell_particle_count;
        N += cell_particle_count;
    }
    return N;
}

/// Generate a density distribution and return the total particle count.
auto generate_varied_density(std::span<uint> particle_count_per_cell) {
    // Target density distribution, split into 9 zones:
    // ┌────────┬──────────────┬────────┐
    // │ 6. mid │ 7.  mid      │ 8. mid │
    // ├────────┼──────────────┼────────┤
    // │ 3. low │ 4. high->low │ 5. mid │
    // ├────────┼──────────────┼────────┤
    // │ 0. mid │ 1.  mid      │ 2. mid │
    // └────────┴──────────────┴────────┘

    auto random_engine = std::default_random_engine(random_seed);
    auto low_distribution = std::uniform_int_distribution<int>(
        0, 4
    );
    auto mid_distribution = std::uniform_int_distribution<int>(
        0.5 * mean_cell_particle_count, 1.5 * mean_cell_particle_count
    );
    auto high_distribution = std::uniform_int_distribution<int>(
        1.5 * mean_cell_particle_count, 2.0 * mean_cell_particle_count
    );
    
    auto N = 0u;

    // Mid distribution (zones 0-2 and 5-8).
    for (auto u = 0; u < U; ++u) {
        for (auto v = 0; v < V; ++v) {
            const auto cell_index = u + v * U;
            // Skip zone 3 and 4.
            if (u < U * 2 / 3 && v >= V / 3 && v < V * 2 / 3) continue;
            const auto cell_particle_count = mid_distribution(random_engine);
            particle_count_per_cell[cell_index] = cell_particle_count;
            N += cell_particle_count;
        }
    }
    // Low distribution (zone 3).
    for (auto u = 0; u < U / 3; ++u) {
        for (auto v = V / 3; v < V * 2 / 3; ++v) {
            const auto cell_index = u + v * U;
            const auto cell_particle_count = low_distribution(random_engine);
            particle_count_per_cell[cell_index] = cell_particle_count;
            N += cell_particle_count;
        }
    }
    // Gradient distribution (zone 4).
    for (auto u = U / 3; u < U * 2 / 3; ++u) {
        for (auto v = V / 3; v < V * 2 / 3; ++v) {
            const auto cell_index = u + v * U;
            const auto mid_contribution = mid_distribution(random_engine);
            const auto high_contribution = high_distribution(random_engine);
            // Left = high, Right = low. Linear gradient between.
            const auto zone_width = U / 3.0;
            const auto weight = (u - U / 3) / zone_width;
            const auto cell_particle_count = weight * mid_contribution
                + (1 - weight) * high_contribution;
            particle_count_per_cell[cell_index] = cell_particle_count;
            N += cell_particle_count;
        }
    }

    return N;
}

/// Randomly distribute particles according with a given density distribution.
void distribute_from_density(
        std::span<FloatingPoint> pos_x,
        std::span<FloatingPoint> pos_y,
        std::span<const uint> particle_indices,
        std::span<const uint> particle_count_per_cell) {
    auto random_engine = std::default_random_engine(random_seed);
#ifdef DEBUG_AVOID_EDGES
    // Make sure no particle is near a cell edge. Will remove edge cases.
    auto distribution_x = std::uniform_real_distribution<FloatingPoint>(
        0.25 * cell.width, 0.75 * cell.width
    );
    auto distribution_y = std::uniform_real_distribution<FloatingPoint>(
        0.25 * cell.height, 0.75 * cell.height
    );
#else
    auto distribution_x = std::uniform_real_distribution<FloatingPoint>(
        0.0, cell.width
    );
    auto distribution_y = std::uniform_real_distribution<FloatingPoint>(
        0.0, cell.height
    );
#endif
    auto cell_index = 0;
    auto indirect_particle_index = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            const auto cell_particle_count = particle_count_per_cell[cell_index];
            for (int i = 0u; i < cell_particle_count; ++i) {
                const auto x = u * cell.width + distribution_x(random_engine) - space.width / 2;
                const auto y = v * cell.height + distribution_y(random_engine) - space.height / 2;
                const auto particle_index = particle_indices[indirect_particle_index];
                pos_x[particle_index] = x;
                pos_y[particle_index] = y;
                ++indirect_particle_index;
            }
            ++cell_index;
        }
    }
}

/// Randomly distribute particles in cells.
void distribute_random(std::span<FloatingPoint> pos_x,
        std::span<FloatingPoint> pos_y) {
    auto random_engine = std::default_random_engine(random_seed);
#ifdef DEBUG_AVOID_EDGES
    // Make sure no particle is near a cell edge. Will remove edge cases.
    auto distribution_x = std::uniform_real_distribution<FloatingPoint>(
        0.25 * cell.width, 0.75 * cell.width
    );
    auto distribution_y = std::uniform_real_distribution<FloatingPoint>(
        0.25 * cell.height, 0.75 * cell.height
    );
#else
    auto distribution_x = std::uniform_real_distribution<FloatingPoint>(
        0.0, cell.width
    );
    auto distribution_y = std::uniform_real_distribution<FloatingPoint>(
        0.0, cell.height
    );
#endif

    auto particle_index = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            for (int i = 0; i < mean_cell_particle_count; ++i) {
                const auto x = u * cell.width + distribution_x(random_engine) - space.width / 2;
                const auto y = v * cell.height + distribution_y(random_engine) - space.height / 2;
                pos_x[particle_index] = x;
                pos_y[particle_index] = y;
                ++particle_index;
            }
        }
    }
}

/// Place all particles in the center of each cell.
void distribute_cell_center(std::span<FloatingPoint> pos_x,
        std::span<FloatingPoint> pos_y) {
    auto particle_index = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            for (int i = 0; i <  mean_cell_particle_count; ++i) {
                const auto x = (u + 0.5) * cell.width - space.width / 2;
                const auto y = (v + 0.5) * cell.height - space.height / 2;
                pos_x[particle_index] = x;
                pos_y[particle_index] = y;
                ++particle_index;
            }
        }
    }
}

void store_positions(std::filesystem::path filepath,
        std::span<const FloatingPoint> x, std::span<const FloatingPoint> y) {
    auto file = std::ofstream(filepath);
    for (auto i = 0; i < x.size(); ++i) {
        file << x[i] << ',';
    }
    file << ";\n";
    for (auto i = 0; i < y.size(); ++i) {
        file << y[i] << ',';
    }
    file << ";\n";
}

void store_density(std::filesystem::path filepath,
        std::span<const FloatingPoint> density) {
    auto file = std::ofstream(filepath);
    for (int row = 0; row < (V + 1); ++row) {
        for (int col = 0; col < (U + 1); ++col) {
            file << density[row * (U + 1) + col] << ',';
        }
        file << '\n';
    }
}

/**
 * Reduce densities using shuffle.
 * 
 * The first thread in each tile will contain the full sum.
 * 
 * @param tile Thread group with static size, where the size 32 is a full warp.
 * @param weights Particle contribution to the density.
 * 
 * @returns um of weights. Only the first thread of the tile holds the full sum.
 */
template<size_t tile_size>
__device__
auto tile_reduce(cooperative_groups::thread_block_tile<tile_size> tile,
        FloatingPoint4 weights) {
    for (auto i = tile.size() / 2; i > 0; i /= 2) {
        weights.x += tile.shfl_down(weights.x, i);
        weights.y += tile.shfl_down(weights.y, i);
        weights.z += tile.shfl_down(weights.z, i);
        weights.w += tile.shfl_down(weights.w, i);
    }
    return weights;
}

#endif
