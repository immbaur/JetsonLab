// Achievable DRAM bandwidth from the GPU: read, write, copy.
// Build: nvcc -O3 -arch=sm_87 -o membw membw.cu
// Usage: ./membw [buffer_MiB=512] [iterations=20]   -> prints one JSON object

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

// 4 independent loads per iteration keep enough requests in flight to saturate DRAM.
__global__ void read_kernel(const float4 *in, size_t n, float *sink) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float a = 0, b = 0, c = 0, d = 0;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        float4 v0 = in[i], v1 = in[i + stride], v2 = in[i + 2 * stride], v3 = in[i + 3 * stride];
        a += v0.x + v0.y + v0.z + v0.w;
        b += v1.x + v1.y + v1.z + v1.w;
        c += v2.x + v2.y + v2.z + v2.w;
        d += v3.x + v3.y + v3.z + v3.w;
    }
    for (; i < n; i += stride) {
        float4 v = in[i];
        a += v.x + v.y + v.z + v.w;
    }
    // Practically never true; keeps the loads from being optimized away.
    if (a + b + c + d == 1234.5f) *sink = a;
}

__global__ void write_kernel(float4 *out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = make_float4(1, 2, 3, 4);
}

__global__ void copy_kernel(const float4 *in, float4 *out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}

template <typename F>
static double best_gbs(F launch, double bytes, int iters) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    launch();  // warmup, also lets the EMC clock ramp up
    CHECK(cudaDeviceSynchronize());
    float best_ms = 1e30f;
    for (int i = 0; i < iters; i++) {
        CHECK(cudaEventRecord(a));
        launch();
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best_ms) best_ms = ms;
    }
    return bytes / (best_ms / 1e3) / 1e9;
}

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? atol(argv[1]) : 512;
    int iters = argc > 2 ? atoi(argv[2]) : 20;
    size_t bytes = mib << 20, n = bytes / sizeof(float4);

    float4 *x, *y;
    float *sink;
    CHECK(cudaMalloc(&x, bytes));
    CHECK(cudaMalloc(&y, bytes));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(x, 0, bytes));

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int blocks = p.multiProcessorCount * 16, threads = 256;

    double read = best_gbs([&] { read_kernel<<<blocks, threads>>>(x, n, sink); }, bytes, iters);
    double write = best_gbs([&] { write_kernel<<<blocks, threads>>>(y, n); }, bytes, iters);
    double copy = best_gbs([&] { copy_kernel<<<blocks, threads>>>(x, y, n); }, 2.0 * bytes, iters);
    double memcpy_d2d = best_gbs([&] { CHECK(cudaMemcpy(y, x, bytes, cudaMemcpyDeviceToDevice)); }, 2.0 * bytes, iters);

    printf("{\"device\": \"%s\", \"buffer_mib\": %zu, \"iterations\": %d, "
           "\"read_gbs\": %.2f, \"write_gbs\": %.2f, \"copy_gbs\": %.2f, \"memcpy_d2d_gbs\": %.2f}\n",
           p.name, mib, iters, read, write, copy, memcpy_d2d);
    return 0;
}
