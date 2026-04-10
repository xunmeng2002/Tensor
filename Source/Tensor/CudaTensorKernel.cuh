#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Tensor.h"

using namespace std;


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
// 检查是否包含零的核函数，注意由Host对result初始化为false
template<typename T>
__global__ void HasZeroKernel(const T* data, size_t elements, bool* result)
{
    __shared__ bool hasZero;
    if (threadIdx.x == 0)
        hasZero = false;
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
// 检查是否相等的核函数，注意由Host对result参数初始化为true
template<typename T>
__global__ void CheckEqualKernel(const T* tensorData, const T* checkData, const int* shape, const int* strides, const int dims, size_t elements, bool* result)
{
    __shared__ bool equal;
    if (threadIdx.x == 0)
        equal = true;
    __syncthreads();

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
    {
        int64_t remaining = index;
        int64_t realOffset = 0;
        for (int64_t d = dims - 1; d >= 0; --d)
        {
            int64_t coord = remaining % shape[d];
            remaining /= shape[d];
            realOffset += coord * strides[d];
        }
        if (tensorData[realOffset] != checkData[index])
            equal = false;
    }
    __syncthreads();

    if (threadIdx.x == 0 && !equal)
    {
        *result = false;
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

template<typename T>
__global__ void CopyFromNoncontiguousKernel(const T* __restrict__ src, T* __restrict__ dst,
    const int* __restrict__ shape, const int* __restrict__ strides, const int dims, const size_t elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements)
        return;
    int linearIndex = idx;
    size_t srcOffset = 0;
    for (int d = dims - 1; d >= 0; --d)
    {
        int coord = linearIndex % shape[d];
        linearIndex /= shape[d];
        srcOffset += coord * strides[d];
    }
    dst[idx] = src[srcOffset];
}

template<typename T>
__global__ void RepeatKernel(const T* srcData, T* destData, const int* srcShape, const int* srcStrides,
    const int* destShape, int64_t src_offset, int dims, int64_t elements)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements)
        return;
	int64_t remaining = index;
    int64_t srcIndex = src_offset;
    for (int64_t d = dims-1; d >= 0; --d)
    {
		int64_t destCoord = remaining % destShape[d];
		remaining /= destShape[d];
		int64_t srcCoord = destCoord % srcShape[d];
        srcIndex += srcCoord * srcStrides[d];
    }
    destData[index] = srcData[srcIndex];
}


