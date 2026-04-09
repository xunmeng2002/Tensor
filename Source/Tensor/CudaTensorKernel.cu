#include "CudaTensorKernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>



void CudaFreeDirect(void* ptr)
{
    cudaFreeAsync(ptr, nullptr);
}




