#ifndef COMMON_HPP
#define COMMON_HPP

#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include <cooperative_groups.h>

//#define DEBUG_DISTRIBUTION
#define DEBUG_AVOID_EDGES

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

const auto cell_particle_count = 16;
#ifndef DEBUG_DISTRIBUTION
// Number of cells in the grid.
const auto U = static_cast<int>(512);
const auto V = static_cast<int>(1024);

#else
const auto U = static_cast<int>(2);
const auto V = static_cast<int>(2);
const int debug_distribution[V][U] = {
    { cell_particle_count, cell_particle_count },
    { cell_particle_count, cell_particle_count }
};
#endif
const auto block_size = 32;

const auto cell_count = U * V;

const auto cell = Dimension{
    .width = space.width / U,
    .height = space.height / V
};

// Density lattice, where each node is a cell corner.
const auto lattice = Dimension{ (U + 1), (V + 1) };
const auto node_count = lattice.width * lattice.height;
// Total number of particles.
#ifndef DEBUG_DISTRIBUTION
const auto N = cell_count * cell_particle_count;
#else
const auto N = ([]{
    auto n = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            n += debug_distribution[v][u];
        }
    }
    return n;
})();
#endif

const auto random_seed = 1u;

const auto warp_size = 32;

template<typename T>
constexpr auto linear_map(const T x, const T x1, const T x2, const T y1, const T y2) {
    const auto slope = (y2 - y1) / (x2 - x1);
    const auto y = (x - x1) * slope + y1;
    return y;
}

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
    return linear_map<FloatingPoint>(x, space.left(), space.right(), 0, U);
}

///Convert world y coordinate to vertical cell index.
constexpr auto y_to_v(FloatingPoint y) {
    return linear_map<FloatingPoint>(y, space.bottom(), space.top(), 0, V);
}

constexpr auto get_node_index(const uint x, const uint y) {
    return x + y * (U + 1);
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
#ifdef DEBUG_DISTRIBUTION
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            for (int i = 0; i < debug_distribution[v][u]; ++i) {
                const auto x = u * cell.width + distribution_x(random_engine) - space.width / 2;
                const auto y = v * cell.height + distribution_y(random_engine) - space.height / 2;
                pos_x[particle_index] = x;
                pos_y[particle_index] = y;
                ++particle_index;
            }
        }
    }
#else
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            for (int i = 0; i < cell_particle_count; ++i) {
                const auto x = u * cell.width + distribution_x(random_engine) - space.width / 2;
                const auto y = v * cell.height + distribution_y(random_engine) - space.height / 2;
                pos_x[particle_index] = x;
                pos_y[particle_index] = y;
                ++particle_index;
            }
        }
    }
#endif
}

/// Place all particles in the center of each cell.
void distribute_cell_center(std::span<FloatingPoint> pos_x,
        std::span<FloatingPoint> pos_y) {
    auto particle_index = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            for (int i = 0; i <  cell_particle_count; ++i) {
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
    for (auto i = 0; i < N; ++i) {
        file << x[i] << ',';
    }
    file << ";\n";
    for (auto i = 0; i < N; ++i) {
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
