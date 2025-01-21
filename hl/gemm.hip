#include <hip/hip_runtime.h>

__global__ void matmulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
