#include "CudaTensorKernel.cuh"


void CudaCheck(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        throw std::runtime_error("CUDA error occurred");
    }
}
void CudaFreeDirect(void* ptr)
{
    cudaFreeAsync(ptr, nullptr);
}
std::shared_ptr<uint8_t> AllocateBytes(size_t bytes)
{
    uint8_t* ptr = nullptr;
    cudaError_t error = cudaMallocAsync(&ptr, bytes, nullptr);
    CUDA_CHECK(error);
    return shared_ptr<uint8_t>(ptr, CudaFreeDirect);
}


