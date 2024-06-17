#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

__global__
void print_point(float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%9.3f, %9.3f\n", x[i], y[i]);
}

int main() {
    // Around 2 million particles.
    const auto N = 1 << 21;

    // Size of 2D space.
    const auto width = 2000.0f;
    const auto height = 4000.0f;
    const auto half_width = width / 2;
    const auto half_height = height / 2;

    const auto random_seed = 1u;

    // Allocation size for 1D arrays.
    const auto size_1d = N * sizeof(float);

    // Allocate particle positions on the host.
    auto h_pos_x = std::vector<float>(size_1d);
    auto h_pos_y = std::vector<float>(size_1d);

    // Allocate particle positions on the device.
    float *d_pos_x;
    float *d_pos_y;
    cudaMalloc(&d_pos_x, size_1d);
    cudaMalloc(&d_pos_y, size_1d);

    // Randomly distribute particles in 2D space.
    auto random_engine = std::default_random_engine(random_seed);
    auto uniform_distribution_x = std::uniform_real_distribution<float>(-half_width, half_width);
    auto uniform_distribution_y = std::uniform_real_distribution<float>(-half_height, half_height);
    for (size_t i = 0; i < size_1d; ++i) {
        h_pos_x[i] = uniform_distribution_x(random_engine);
        h_pos_y[i] = uniform_distribution_y(random_engine);
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(10) << h_pos_x[i]
                  << std::setw(10) << h_pos_y[i]
                  << '\n';
    }
    std::cout << '\n';

    // Copy positions from the host to the device.
    cudaMemcpy(d_pos_x, h_pos_x.data(), size_1d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y.data(), size_1d, cudaMemcpyHostToDevice);

    print_point<<<1, 10>>>(d_pos_x, d_pos_y);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
}
