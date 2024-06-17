#include <memory>

int main() {
    // Around 2 million particles.
    const int N = 1 << 21;

    // Allocation size for 1D arrays.
    auto size_1d = N * sizeof(float);

    // Allocate particle positions on the host.
    auto h_pos_x = std::make_unique<float>(size_1d);
    auto h_pos_y = std::make_unique<float>(size_1d);

    // Allocate particle positions on the device.
    float *d_pos_x;
    float *d_pos_y;
    cudaMalloc(&d_pos_x, size_1d);
    cudaMalloc(&d_pos_y, size_1d);

    // Free device memory.
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
}
