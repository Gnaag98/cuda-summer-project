#ifndef COMMON_HPP
#define COMMON_HPP

#include <random>
#include <span>

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
const auto space = Rectangle<float>::centered(2048.0f, 4096.0f);

// Number of cells in the grid.
const auto U = static_cast<int>(128);
const auto V = static_cast<int>(256);

const auto cell = Dimension{
    .width = space.width / U,
    .height = space.height / V
};

// Lattice node count, where each node is a cell corner.
const auto node_count = Dimension{ (U + 1), (V + 1) };

const int cell_particle_count = 1024;

// Total number of particles.
const auto N = cell_particle_count * U * V;

const auto random_seed = 1u;

// Allocation size for 1D arrays.
const auto positions_bytes = N * sizeof(float);
const auto lattice_bytes = (U + 1) * (V + 1) * sizeof(float);

template<typename T>
constexpr auto linear_map(const T x, const T x1, const T x2, const T y1, const T y2) {
    const auto slope = (y2 - y1) / (x2 - x1);
    const auto y = (x - x1) * slope + y1;
    return y;
}

/**
 * Convert world x coordinate to horizontal cell index.
 */
constexpr auto x_to_u(float x) {
    return linear_map<float>(x, space.left(), space.right(), 0, U);
}

/**
 * Convert world y coordinate to vertical cell index.
 */
constexpr auto y_to_v(float y) {
    return linear_map<float>(y, space.bottom(), space.top(), 0, V);
}

constexpr auto get_node_index(const int x, const int y) {
    return x + y * (U + 1);
}

constexpr auto get_particle_index(const int i, const int u, const int v) {
    return i + (u + v * U) * cell_particle_count;
}

void distribute_random(std::span<float> pos_x, std::span<float> pos_y) {
    // Randomly distribute particles in cells.
    auto random_engine = std::default_random_engine(random_seed);
    auto distribution_x = std::uniform_real_distribution(
        0.0f, cell.width
    );
    auto distribution_y = std::uniform_real_distribution(
        0.0f, cell.height
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

#endif
