#ifndef COMMON_HPP
#define COMMON_HPP

#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <type_traits>

#include <cooperative_groups.h>

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

// Size of 2D space.
const auto space = Rectangle<FloatingPoint>::centered(2048.0, 4096.0);

// Number of cells in the grid.
const auto U = static_cast<int>(128);
const auto V = static_cast<int>(256);

const auto cell = Dimension{
    .width = space.width / U,
    .height = space.height / V
};

// Lattice node count, where each node is a cell corner.
const auto node_count = Dimension{ (U + 1), (V + 1) };

const auto cell_particle_count = 1024;

// Total number of particles.
const auto N = cell_particle_count * U * V;

const auto random_seed = 1u;

// Allocation size for 1D arrays.
const auto positions_count = N;
const auto lattice_count = (U + 1) * (V + 1);
const auto positions_bytes = positions_count * sizeof(FloatingPoint);
const auto indices_bytes = positions_count * sizeof(size_t);
const auto lattice_bytes = lattice_count * sizeof(FloatingPoint);

const auto block_size = 128;
const auto blocks_per_cell = (cell_particle_count + block_size - 1) / block_size;
const auto block_count = U * V * blocks_per_cell;

const auto warp_size = 32;

template<typename T>
constexpr auto linear_map(const T x, const T x1, const T x2, const T y1, const T y2) {
    const auto slope = (y2 - y1) / (x2 - x1);
    const auto y = (x - x1) * slope + y1;
    return y;
}

/**
 * Convert world x coordinate to horizontal cell index.
 */
constexpr auto x_to_u(FloatingPoint x) {
    return linear_map<FloatingPoint>(x, space.left(), space.right(), 0, U);
}

/**
 * Convert world y coordinate to vertical cell index.
 */
constexpr auto y_to_v(FloatingPoint y) {
    return linear_map<FloatingPoint>(y, space.bottom(), space.top(), 0, V);
}

constexpr auto get_node_index(const int x, const int y) {
    return x + y * (U + 1);
}

constexpr auto get_particle_index(const int i, const int u, const int v) {
    return i + (u + v * U) * cell_particle_count;
}

void distribute_random(std::span<FloatingPoint> pos_x, std::span<FloatingPoint> pos_y) {
    // Randomly distribute particles in cells.
    auto random_engine = std::default_random_engine(random_seed);
    auto distribution_x = std::uniform_real_distribution<FloatingPoint>(
        0.0, cell.width
    );
    auto distribution_y = std::uniform_real_distribution<FloatingPoint>(
        0.0, cell.height
    );

    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            for (int i = 0; i <  cell_particle_count; ++i) {
                const auto particle_index = get_particle_index(i, u, v);
                const auto x = u * cell.width + distribution_x(random_engine) - space.width / 2;
                const auto y = v * cell.height + distribution_y(random_engine) - space.height / 2;
                pos_x[particle_index] = x;
                pos_y[particle_index] = y;
            }
        }
    }
}

void distribute_cell_center(std::span<FloatingPoint> pos_x, std::span<FloatingPoint> pos_y) {
    // Place all particles in the center of each cell.
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            for (int i = 0; i <  cell_particle_count; ++i) {
                const auto particle_index = get_particle_index(i, u, v);
                const auto x = (u + 0.5) * cell.width - space.width / 2;
                const auto y = (v + 0.5) * cell.height - space.height / 2;
                pos_x[particle_index] = x;
                pos_y[particle_index] = y;
            }
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
