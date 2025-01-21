#include "gemm.hpp"
#include <iostream>

#define HIP_CALL(call) do {                                  \
    hipError_t err = call;                                   \
    if (err != hipSuccess) {                                 \
        std::cerr << "HIP error at " << __FILE__ << ":"      \
                  << __LINE__ << " - " << hipGetErrorString(err) \
                  << std::endl;                              \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)


int main() {
  int device_id;
  HIP_CALL(hipGetDevice(&device_id));

  hipDeviceProp_t deviceProp;
  HIP_CALL(hipGetDeviceProperties(&deviceProp, device_id));

  std::cout << "Using device: " << deviceProp.name << std::endl;
  // Matrix dimensions:
  //   A: M x K
  //   B: K x N
  //   C: M x N
  const int M = 256;
  const int K = 256;
  const int N = 256;

  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  // Allocate and initialize host matrices
  float* h_A = new float[M * K];
  float* h_B = new float[K * N];
  float* h_C = new float[M * N];

  // Initialize A and B with some values
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<float>(i % 100) / 100.0f; // values between 0 and 0.99
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<float>((i + 1) % 100) / 100.0f; // values between 0 and 0.99
  }

  // Allocate device memory
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  HIP_CALL(hipMalloc(&d_A, sizeA));
  HIP_CALL(hipMalloc(&d_B, sizeB));
  HIP_CALL(hipMalloc(&d_C, sizeC));

  // Copy host memory to device
  HIP_CALL(hipMemcpy(d_A, h_A, sizeA, hipMemcpyHostToDevice));
  HIP_CALL(hipMemcpy(d_B, h_B, sizeB, hipMemcpyHostToDevice));

  // Set up execution configuration
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the kernel
  hipLaunchKernelGGL(matmulKernel, blocksPerGrid, threadsPerBlock, 0, 0,
      d_A, d_B, d_C, M, N, K);
  HIP_CALL(hipDeviceSynchronize());

  // Copy the result back to the host
  HIP_CALL(hipMemcpy(h_C, d_C, sizeC, hipMemcpyDeviceToHost));

  // (Optional) Validate or print a few results from the output matrix
  std::cout << "Sample output values:" << std::endl;
  std::cout << "C[0]: " << h_C[0] << std::endl;
  std::cout << "C[mid]: " << h_C[(M * N) / 2] << std::endl;
  std::cout << "C[last]: " << h_C[M * N - 1] << std::endl;

  // Free device and host memory
  HIP_CALL(hipFree(d_A));
  HIP_CALL(hipFree(d_B));
  HIP_CALL(hipFree(d_C));

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  std::cout << "Matrix multiplication completed successfully." << std::endl;
  return 0;
  return 0;
}
