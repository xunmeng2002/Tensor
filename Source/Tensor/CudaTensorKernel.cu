#include "CudaTensorKernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>



void CudaFreeNone(void* ptr)
{
}
void CudaFreeDirect(void* ptr)
{
    cudaFreeAsync(ptr, nullptr);
}
void CudaFreePool(void* ptr)
{
}

void* AllocateKernel(size_t size)
{
    char* ptr = nullptr;
    cudaError_t error = cudaMallocAsync(&ptr, size, nullptr);
    CUDA_CHECK(error);
    return ptr;
}





