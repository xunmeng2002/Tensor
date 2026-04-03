#pragma once
#include <cuda_runtime.h>
#include "Tensor.h"

using namespace std;

#define CUDA_CHECK(error) \
    do { \
        cudaError_t err = error; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error occurred"); \
        } \
    } while(0)


void CudaFreeNone(void* ptr);
void CudaFreeDirect(void* ptr);
void CudaFreePool(void* ptr);

void* AllocateKernel(size_t size);

// 元素级运算核函数
template<typename T>
__global__ void ElementwiseKernel(const T * a, const T * b, T * c, size_t elements, int op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements)
        return;

    switch (op)
    {
    case 0: c[idx] = a[idx] + b[idx]; break;  // ADD
    case 1: c[idx] = a[idx] - b[idx]; break;  // SUB
    case 2: c[idx] = a[idx] * b[idx]; break;  // MUL
    case 3: c[idx] = a[idx] / b[idx]; break;  // DIV
    }
}
// 原地运算核函数
template<typename T>
__global__ void ElementwiseInplaceKernel(T* a, const T* b, size_t elements, int op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements)
        return;

    switch (op)
    {
    case 0: a[idx] += b[idx]; break;
    case 1: a[idx] -= b[idx]; break;
    case 2: a[idx] *= b[idx]; break;
    case 3: a[idx] /= b[idx]; break;
    }
}
// 标量运算核函数
template<typename T>
__global__ void ScalarKernel(const T* a, T* c, size_t elements, T scalar, int op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements)
        return;

    switch (op)
    {
    case 0: c[idx] = a[idx] + scalar; break;
    case 1: c[idx] = a[idx] - scalar; break;
    case 2: c[idx] = a[idx] * scalar; break;
    case 3: c[idx] = a[idx] / scalar; break;
    }
}
// 标量原地运算核函数
template<typename T>
__global__ void ScalarInplaceKernel(T* a, size_t elements, T scalar, int op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements)
        return;

    switch (op)
    {
    case 0: a[idx] += scalar; break;
    case 1: a[idx] -= scalar; break;
    case 2: a[idx] *= scalar; break;
    case 3: a[idx] /= scalar; break;
    }
}
// 检查是否包含零的核函数
template<typename T>
__global__ void HasZeroKernel(const T* data, size_t elements, bool* result)
{
    __shared__ bool hasZero;
    if (threadIdx.x == 0) hasZero = false;
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements && data[idx] == T(0)) {
        hasZero = true;
    }
    __syncthreads();

    if (threadIdx.x == 0 && hasZero) {
        *result = true;
    }
}
// 填充核函数
template<typename T>
__global__ void FillKernel(T* data, size_t elements, T value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        data[idx] = value;
    }
}
// 等差数列填充核函数
template<typename T>
__global__ void ArrangeKernel(T* data, size_t elements, T start, T step)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        data[idx] = start + static_cast<T>(idx) * step;
    }
}




// 核函数启动接口
template<typename T>
void LaunchElementwiseKernel(const T* a, const T* b, T* c, size_t elements, int blockSize, ElementwiseOp op)
{
    int gridSize = (elements + blockSize - 1) / blockSize;

    ElementwiseKernel<T><<<gridSize, blockSize>>>(a, b, c, elements, static_cast<int>(op));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void LaunchElementwiseKernelInplace(T* a, const T* b, size_t elements, int blockSize, ElementwiseOp op)
{
    int gridSize = (elements + blockSize - 1) / blockSize;

    ElementwiseInplaceKernel<T> << <gridSize, blockSize >> > (a, b, elements, static_cast<int>(op));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void LaunchScalarKernel(const T* a, T* c, size_t elements, T scalar, int blockSize, ScalarOp op)
{
    int gridSize = (elements + blockSize - 1) / blockSize;

    ScalarKernel<T> << <gridSize, blockSize >> > (a, c, elements, scalar, static_cast<int>(op));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void LaunchScalarKernelInplace(T* a, size_t elements, T scalar, int blockSize, ScalarOp op)
{
    int gridSize = (elements + blockSize - 1) / blockSize;

    ScalarInplaceKernel<T> << <gridSize, blockSize >> > (a, elements, scalar, static_cast<int>(op));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
bool LaunchHasZeroKernel(const T* data, size_t elements, int blockSize)
{
    bool* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(bool)));

    int threads = 256;
    int blocks = (elements + threads - 1) / threads;

    HasZeroKernel<T> << <blocks, threads >> > (data, elements, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

template<typename T>
void LaunchFillKernel(T* data, size_t elements, T value, int blockSize)
{
    int gridSize = (elements + blockSize - 1) / blockSize;

    FillKernel<T> << <gridSize, blockSize >> > (data, elements, value);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void LaunchArrangeKernel(T* data, size_t elements, T start, T step, int blockSize)
{
    int gridSize = (elements + blockSize - 1) / blockSize;

    ArrangeKernel<T> << <gridSize, blockSize >> > (data, elements, start, step);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}



