#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

// Around 2 million particles.
const auto N = 1 << 21;

// Switch between random and grid distribution.
const auto is_randomly_distributed = true;

// Size of 2D space.
const auto width = 2000.0f;
const auto height = 4000.0f;
const auto half_width = width / 2;
const auto half_height = height / 2;

// Size of 2D density grid.
const auto U = static_cast<int>(width / 20);
const auto V = static_cast<int>(height / 20);

const auto random_seed = 1u;

// Allocation size for 1D arrays.
const auto position_size = N * sizeof(float);
const auto density_size = U * V * sizeof(float);

__host__ __device__
auto x_to_u(float x) {
    return (x / width + 0.5f) * (U - 1);
}

__host__ __device__
auto y_to_v(float y) {
    return (y / height + 0.5f) * (V - 1);
}

__host__ __device__
auto get_density_index(const int2 uv) {
    return uv.y * U + uv.x;
}

__global__
void print_point(float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("(%9.3f, %9.3f)\n", x[i], y[i]);
}

__global__
void add_density_atomic(float *pos_x, float *pos_y, float *density) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return;
    }

    const auto x = pos_x[index];
    const auto y = pos_y[index];
    // Add 0.5 to move from [-0.5, 0.5) to [0, 1).
    const auto u = x_to_u(x);
    const auto v = y_to_v(y);
    //printf("(%9.3f, %9.3f) -> (%7.3f, %7.3f)\n", x, y, u, v);

    // Node coordinates.
    const auto pos_bottom_left = int2{ static_cast<int>(floor(u)), static_cast<int>(floor(v)) };
    const auto pos_bottom_right = int2{ static_cast<int>(ceil(u)), static_cast<int>(floor(v)) };
    const auto pos_top_left = int2{ static_cast<int>(floor(u)), static_cast<int>(ceil(v)) };
    const auto pos_top_right = int2{ static_cast<int>(ceil(u)), static_cast<int>(ceil(v)) };

    // Node weights. https://www.particleincell.com/2010/es-pic-method/
    const auto pos_cell = float2{ u - pos_bottom_left.x, v - pos_bottom_left.y };
    const auto weight_bottom_left  = (1 - pos_cell.x) * (1 - pos_cell.y);
    const auto weight_bottom_right =      pos_cell.x  * (1 - pos_cell.y);
    const auto weight_top_left     = (1 - pos_cell.x) *      pos_cell.y;
    const auto weight_top_right    =      pos_cell.x  *      pos_cell.y;

    // Node indices
    const auto index_bottom_left = get_density_index(pos_bottom_left);
    const auto index_bottom_right = get_density_index(pos_bottom_right);
    const auto index_top_left = get_density_index(pos_top_left);
    const auto index_top_right = get_density_index(pos_top_right);

    atomicAdd(&density[index_bottom_left], weight_bottom_left);
    atomicAdd(&density[index_bottom_right], weight_bottom_right);
    atomicAdd(&density[index_top_left], weight_top_left);
    atomicAdd(&density[index_top_right], weight_top_right);

    /* printf("\n(%d, %d), (%d, %d)\n", pos_top_left.x, pos_top_left.y, pos_top_right.x, pos_top_right.y);
    printf("(%d, %d), (%d, %d)\n", pos_bottom_left.x, pos_bottom_left.y, pos_bottom_right.x, pos_bottom_right.y);

    printf("%d, %d\n", index_top_left, index_top_right);
    printf("%d, %d\n", index_bottom_left, index_bottom_right);

    printf("%7.3f, %7.3f\n", weight_top_left, weight_top_right);
    printf("%7.3f, %7.3f\n\n", weight_bottom_left, weight_bottom_right); */
}

void distribute_random(std::span<float> pos_x, std::span<float> pos_y) {
    // Randomly distribute particles in 2D space.
    auto random_engine = std::default_random_engine(random_seed);
    auto uniform_distribution_x = std::uniform_real_distribution<float>(-half_width, half_width);
    auto uniform_distribution_y = std::uniform_real_distribution<float>(-half_height, half_height);
    for (size_t i = 0; i < N; ++i) {
        pos_x[i] = uniform_distribution_x(random_engine);
        pos_y[i] = uniform_distribution_y(random_engine);
    }
}

void distribute_grid(std::span<float> pos_x, std::span<float> pos_y) {
    // Has to be a power of two to easily factor N into rows and columns.
    auto is_power_of_two = [](const unsigned long x) {
        // https://stackoverflow.com/a/600306
        return (x != 0) && ((x & (x - 1)) == 0);
    };
    if (!is_power_of_two(N)) {
        throw std::runtime_error("N has to be a power of 2 for grid distribution.");
    }

    // Factor N so that rows * columns = N.
    const auto power = static_cast<int>(log2(N));
    const auto row_power = power / 2;
    const auto column_power = power - row_power;
    const auto rows = 1 << row_power;
    const auto columns = 1 << column_power;
    
    // Iterate over all N = rows * columns particles.
    for (size_t j = 0; j < rows; ++j) {
        for (size_t i = 0; i < columns; ++i) {
            pos_x[j * columns + i] = i * width / columns - half_width;
            pos_y[j * columns + i] = j * height / rows - half_height;
        }
    }
}

int main() {
    // Allocate particle positions and densities on the host.
    auto h_pos_x = std::vector<float>(position_size);
    auto h_pos_y = std::vector<float>(position_size);
    auto h_density = std::vector<float>(density_size);

    // Allocate particle positions and densities on the device.
    float *d_pos_x;
    float *d_pos_y;
    float *d_density;
    cudaMalloc(&d_pos_x, position_size);
    cudaMalloc(&d_pos_y, position_size);
    cudaMalloc(&d_density, density_size);

    if (is_randomly_distributed) {
        distribute_random(h_pos_x, h_pos_y);
    } else {
        distribute_grid(h_pos_x, h_pos_y);
    }

    // Copy positions from the host to the device.
    cudaMemcpy(d_pos_x, h_pos_x.data(), position_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y.data(), position_size, cudaMemcpyHostToDevice);

    // Compute the particle density.
    const int block_size = 256;
    const int block_count = (N + block_size - 1) / block_size;
    add_density_atomic<<<block_count, block_size>>>(d_pos_x, d_pos_y, d_density);
    cudaDeviceSynchronize();
    cudaMemcpy(h_density.data(), d_density, density_size, cudaMemcpyDeviceToHost);

    // Store data to files.
    auto density_file = std::ofstream("density.csv");
    for (int row = 0; row < V; ++row) {
        for (int col = 0; col < U; ++col) {
            density_file << h_density[row * U + col] << ',';
        }
        density_file << '\n';
    }
    auto positions_file = std::ofstream("positions.csv");
    for (int i = 0; i < N; ++i) {
        positions_file << h_pos_x[i] << ',' << h_pos_y[i] << '\n';
    }

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
}
