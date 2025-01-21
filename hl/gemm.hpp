#pragma once
#include <hip/hip_runtime.h>

__global__ void matmulKernel(const float* A, const float* B, float* C, int M, int N, int K);
