#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Around 2 million particles.
const auto N = 1 << 21;

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

__global__
void print_point(float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("(%9.3f, %9.3f)\n", x[i], y[i]);
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

    // Randomly distribute particles in 2D space.
    auto random_engine = std::default_random_engine(random_seed);
    auto uniform_distribution_x = std::uniform_real_distribution<float>(-half_width, half_width);
    auto uniform_distribution_y = std::uniform_real_distribution<float>(-half_height, half_height);
    for (size_t i = 0; i < position_size; ++i) {
        h_pos_x[i] = uniform_distribution_x(random_engine);
        h_pos_y[i] = uniform_distribution_y(random_engine);
    }

    // DEBUG: Print 10 host points.
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(10) << h_pos_x[i]
                  << std::setw(10) << h_pos_y[i]
                  << '\n';
    }
    std::cout << '\n';

    // Copy positions from the host to the device.
    cudaMemcpy(d_pos_x, h_pos_x.data(), position_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y.data(), position_size, cudaMemcpyHostToDevice);

    // DEBUG: Print 10 device points.
    print_point<<<1, 10>>>(d_pos_x, d_pos_y);
    cudaDeviceSynchronize();
    std::cout << '\n';

    print_cell_origin<<<1, 1>>>(d_pos_x, d_pos_y, d_density);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
}
