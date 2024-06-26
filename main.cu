#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

template<typename T>
struct Vec3 {
    T x;
    T y;
    T z;

    __host__ __device__
    constexpr static Vec3 cross(const Vec3 &a, const Vec3 &b) {
        return Vec3{
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
};

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

    __host__ __device__
    constexpr T left() const { return x; }
    __host__ __device__
    constexpr T right() const { return x + width; }
    __host__ __device__
    constexpr T bottom() const { return y; }
    __host__ __device__
    constexpr T top() const { return y + height; }
};

// Switch between random and grid distribution.
const auto is_randomly_distributed = true;

// Number of kernel iterations. Higher = better for profiling.
const auto kernel_iteration_count = 100;

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
const auto node_count = int2{ (U + 1), (V + 1) };

const int cell_particle_count = 1024;

// Total number of particles.
const auto N = cell_particle_count * U * V;

const auto random_seed = 1u;

// Allocation size for 1D arrays.
const auto positions_bytes = N * sizeof(float);
const auto lattice_bytes = node_count.x * node_count.y * sizeof(float);

template<typename T>
__host__ __device__
auto linear_map(const T x, const T x1, const T x2, const T y1, const T y2) {
    const auto slope = (y2 - y1) / (x2 - x1);
    const auto y = (x - x1) * slope + y1;
    return y;
}

/**
 * Convert world x coordinate to horizontal cell index.
 */
__host__ __device__
auto x_to_u(float x) {
    return linear_map<float>(x, space.left(), space.right(), 0, U);
}

/**
 * Convert world y coordinate to vertical cell index.
 */
__host__ __device__
auto y_to_v(float y) {
    return linear_map<float>(y, space.bottom(), space.top(), 0, V);
}

__host__ __device__
auto get_node_index(const int2 node) {
    return node.x + node.y * node_count.x;
}

__host__ __device__
auto get_particle_index(const int i, const int u, const int v) {
    return i + (u + v * U) * cell_particle_count;
}

__global__
void add_density_atomic(float *const pos_x, float *const pos_y, float *density) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return;
    }

    const auto x = pos_x[index];
    const auto y = pos_y[index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);
    //printf("(%9.3f, %9.3f) -> (%7.3f, %7.3f)\n", x, y, u, v);

    // Node coordinates.
    const auto node_bottom_left  = int2{ static_cast<int>(floor(u)), static_cast<int>(floor(v)) };
    const auto node_bottom_right = int2{ static_cast<int>( ceil(u)), static_cast<int>(floor(v)) };
    const auto node_top_left     = int2{ static_cast<int>(floor(u)), static_cast<int>( ceil(v)) };
    const auto node_top_right    = int2{ static_cast<int>( ceil(u)), static_cast<int>( ceil(v)) };

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_relative_cell = float2{ u - node_bottom_left.x, v - node_bottom_left.y };
    const auto weight_bottom_left  = (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y);
    const auto weight_bottom_right =      pos_relative_cell.x  * (1 - pos_relative_cell.y);
    const auto weight_top_left     = (1 - pos_relative_cell.x) *      pos_relative_cell.y;
    const auto weight_top_right    =      pos_relative_cell.x  *      pos_relative_cell.y;

    // Node indices.
    const auto index_bottom_left = get_node_index(node_bottom_left);
    const auto index_bottom_right = get_node_index(node_bottom_right);
    const auto index_top_left = get_node_index(node_top_left);
    const auto index_top_right = get_node_index(node_top_right);

    atomicAdd(&density[index_bottom_left], weight_bottom_left);
    atomicAdd(&density[index_bottom_right], weight_bottom_right);
    atomicAdd(&density[index_top_left], weight_top_left);
    atomicAdd(&density[index_top_right], weight_top_right);
}

__global__
void add_density_dummy(float *const pos_x, float *const pos_y) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return;
    }

    const auto x = pos_x[index];
    const auto y = pos_y[index];
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);
    //printf("(%9.3f, %9.3f) -> (%7.3f, %7.3f)\n", x, y, u, v);

    // Node coordinates.
    const auto node_bottom_left  = int2{ static_cast<int>(floor(u)), static_cast<int>(floor(v)) };
    const auto node_bottom_right = int2{ static_cast<int>( ceil(u)), static_cast<int>(floor(v)) };
    const auto node_top_left     = int2{ static_cast<int>(floor(u)), static_cast<int>( ceil(v)) };
    const auto node_top_right    = int2{ static_cast<int>( ceil(u)), static_cast<int>( ceil(v)) };

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_relative_cell = float2{ u - node_bottom_left.x, v - node_bottom_left.y };
    const auto weight_bottom_left  = (1 - pos_relative_cell.x) * (1 - pos_relative_cell.y);
    const auto weight_bottom_right =      pos_relative_cell.x  * (1 - pos_relative_cell.y);
    const auto weight_top_left     = (1 - pos_relative_cell.x) *      pos_relative_cell.y;
    const auto weight_top_right    =      pos_relative_cell.x  *      pos_relative_cell.y;

    // Node indices.
    const auto index_bottom_left = get_node_index(node_bottom_left);
    const auto index_bottom_right = get_node_index(node_bottom_right);
    const auto index_top_left = get_node_index(node_top_left);
    const auto index_top_right = get_node_index(node_top_right);

    auto dummy_sum = 0.0f;
    dummy_sum += weight_bottom_left;
    dummy_sum += weight_bottom_right;
    dummy_sum += weight_top_left;
    dummy_sum += weight_top_right;
}

__global__
void calculate_gradient(float *const pos_x, float *const pos_y, float *const density) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return;
    }

    auto lerp = [](const float a, const float b, const float t) {
        return a + (b - a) * t;
    };

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
    const auto index_bottom_left = get_node_index(node_bottom_left);
    const auto index_bottom_right = get_node_index(node_bottom_right);
    const auto index_top_left = get_node_index(node_top_left);
    const auto index_top_right = get_node_index(node_top_right);

    const auto density_bottom_left = density[index_bottom_left];
    const auto density_bottom_right = density[index_bottom_right];
    const auto density_top_left = density[index_top_left];
    const auto density_top_right = density[index_top_right];

    const auto pos_relative_cell = float2{ u - node_bottom_left.x, v - node_bottom_left.y };

    const auto density_bottom = lerp(density_bottom_left, density_bottom_right, pos_relative_cell.x);
    const auto density_top = lerp(density_top_left, density_top_right, pos_relative_cell.x);
    [[maybe_unused]] const auto slope_y = (density_top - density_bottom) / space.height;

    const auto density_left = lerp(density_bottom_left, density_top_left, pos_relative_cell.y);
    const auto density_right = lerp(density_bottom_right, density_top_right, pos_relative_cell.y);
    [[maybe_unused]] const auto slope_x = (density_right - density_left) / space.width;

    const auto tangent_x = Vec3<float>{
        .x = space.width,
        .y = 0,
        .z = density_right - density_left
    };
    const auto tangent_y = Vec3<float>{
        .x = 0,
        .y = space.height,
        .z = density_top - density_bottom
    };
    const auto normal = Vec3<float>::cross(tangent_x, tangent_y);
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

void distribute_cell_center(std::span<float> pos_x, std::span<float> pos_y) {
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
                   std::span<const float> density) {
    auto density_file = std::ofstream(filepath);
    for (int row = 0; row < node_count.y; ++row) {
        for (int col = 0; col < node_count.x; ++col) {
            density_file << density[row * node_count.x + col] << ',';
        }
        density_file << '\n';
    }
}

void store_positions(std::filesystem::path filepath,
                     std::span<const float> pos_x,
                     std::span<const float> pos_y) {
    auto positions_file = std::ofstream(filepath);
    for (int i = 0; i < N; ++i) {
        positions_file << pos_x[i] << ',' << pos_y[i] << '\n';
    }
}

int main() {
    // Allocate particle positions and densities on the host.
    auto h_pos_x = std::vector<float>(positions_bytes);
    auto h_pos_y = std::vector<float>(positions_bytes);
    auto h_density = std::vector<float>(lattice_bytes);

    // Allocate particle positions and densities on the device.
    float *d_pos_x;
    float *d_pos_y;
    float *d_density;
    cudaMalloc(&d_pos_x, positions_bytes);
    cudaMalloc(&d_pos_y, positions_bytes);
    cudaMalloc(&d_density, lattice_bytes);

    if (is_randomly_distributed) {
        distribute_random(h_pos_x, h_pos_y);
    } else {
        distribute_cell_center(h_pos_x, h_pos_y);
    }

    // Copy positions from the host to the device.
    cudaMemcpy(d_pos_x, h_pos_x.data(), positions_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y.data(), positions_bytes, cudaMemcpyHostToDevice);

    // Compute the particle density.
    const auto block_size = 256;
    const auto block_count = (N + block_size - 1) / block_size;

    // Loop for better statistics.
    for (int i = 0; i < kernel_iteration_count; ++i) {
        add_density_atomic<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_density);
        cudaDeviceSynchronize();
        cudaMemcpy(h_density.data(), d_density, lattice_bytes, cudaMemcpyDeviceToHost);

        // Dummy calculations.
        add_density_dummy<<<block_count, block_size>>>(d_pos_x, d_pos_y);
        cudaDeviceSynchronize();
        calculate_gradient<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_density);
        cudaDeviceSynchronize();
    }

    // Store data to files.
    const auto output_directory = std::filesystem::path("output");
    std::filesystem::create_directory(output_directory);
    store_density(output_directory / "density.csv", h_density);
    store_positions(output_directory / "positions.csv", h_pos_x, h_pos_y);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
}
