#pragma once
#include "CudaTensor.h"
#include "CudaTensorKernel.cuh"



// 算术运算实现
template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Add(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        // 检查是否需要广播
        std::vector<int> broadcastShape;
        CheckAndBroadcast(other, broadcastShape);
        if (broadcastShape.empty())
            return nullptr;
        return BroadcastOp(other, "add");
    }

    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchElementwiseKernel(
        this->pdata.get() + this->offset,
        other.GetData().get() + other.GetOffset(),
        result->GetData().get(),
        elements,
        defaultBlockSize,
        ElementwiseOp::ADD
    );

    return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Sub(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        std::vector<int> broadcastShape;
        CheckAndBroadcast(other, broadcastShape);
        if (broadcastShape.empty())
            return nullptr;
        return BroadcastOp(other, "sub");
    }

    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchElementwiseKernel(
        this->pdata.get() + this->offset,
		other.GetData().get() + other.GetOffset(),
        result->GetData().get(),
        elements,
        defaultBlockSize,
        ElementwiseOp::SUB
    );

    return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Mul(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        std::vector<int> broadcastShape;
        CheckAndBroadcast(other, broadcastShape);
        if (broadcastShape.empty())
            return nullptr;
        return BroadcastOp(other, "mul");
    }

    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchElementwiseKernel(
        this->pdata.get() + this->offset,
        other.GetData().get() + other.GetOffset(),
        result->GetData().get(),
        elements,
        defaultBlockSize,
        ElementwiseOp::MUL
    );

    return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Div(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        std::vector<int> broadcastShape;
        CheckAndBroadcast(other, broadcastShape);
        if (broadcastShape.empty())
            return nullptr;
        return BroadcastOp(other, "div");
    }

    if (other.HasZero())
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
    }

    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchElementwiseKernel(
        this->pdata.get() + this->offset,
        other.GetData().get() + other.GetOffset(),
        result->GetData().get(),
        elements,
        defaultBlockSize,
        ElementwiseOp::DIV
    );

    return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Add(T value) const
{
    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchScalarKernel(
        this->pdata.get() + this->offset,
        result->GetData().get(),
        elements,
        value,
        defaultBlockSize,
        ScalarOp::ADD
    );

    return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Sub(T value) const
{
    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchScalarKernel(
        this->pdata.get() + this->offset,
        result->GetData().get(),
        elements,
        value,
        defaultBlockSize,
        ScalarOp::SUB
    );

    return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Mul(T value) const
{
    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchScalarKernel(
        this->pdata.get() + this->offset,
        result->GetData().get(),
        elements,
        value,
        defaultBlockSize,
        ScalarOp::MUL
    );

    return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Div(T value) const
{
    if (value == T(0))
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
    }

    auto result = Create(this->shape);
    size_t elements = this->GetTotalElements();

    LaunchScalarKernel(
        this->pdata.get() + this->offset,
        result->GetData().get(),
        elements,
        value,
        defaultBlockSize,
        ScalarOp::DIV
    );

    return result;
}

// 原地运算实现
template<typename T>
void CudaTensor<T>::Add_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
        return;

    size_t elements = this->GetTotalElements();
    LaunchElementwiseKernelInplace(
        this->pdata.get() + this->offset,
        other.GetData().get() + other.GetOffset(),
        elements,
        defaultBlockSize,
        ElementwiseOp::ADD
    );
}

template<typename T>
void CudaTensor<T>::Sub_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
        return;

    size_t elements = this->GetTotalElements();
    LaunchElementwiseKernelInplace(
        this->pdata.get() + this->offset,
        other.GetData().get() + other.GetOffset(),
        elements,
        defaultBlockSize,
        ElementwiseOp::SUB
    );
}

template<typename T>
void CudaTensor<T>::Mul_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
        return;

    size_t elements = this->GetTotalElements();
    LaunchElementwiseKernelInplace(
        this->pdata.get() + this->offset,
        other.GetData().get() + other.GetOffset(),
        elements,
        defaultBlockSize,
        ElementwiseOp::MUL
    );
}

template<typename T>
void CudaTensor<T>::Div_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
        return;

    if (other.HasZero())
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
    }

    size_t elements = this->GetTotalElements();
    LaunchElementwiseKernelInplace(
        this->pdata.get() + this->offset,
        other.GetData().get() + other.GetOffset(),
        elements,
        defaultBlockSize,
        ElementwiseOp::DIV
    );
}

template<typename T>
void CudaTensor<T>::Add_(T value)
{
    size_t elements = this->GetTotalElements();
    LaunchScalarKernelInplace(
        this->pdata.get() + this->offset,
        elements,
        value,
        defaultBlockSize,
        ScalarOp::ADD
    );
}

template<typename T>
void CudaTensor<T>::Sub_(T value)
{
    size_t elements = this->GetTotalElements();
    LaunchScalarKernelInplace(
        this->pdata.get() + this->offset,
        elements,
        value,
        defaultBlockSize,
        ScalarOp::SUB
    );
}

template<typename T>
void CudaTensor<T>::Mul_(T value)
{
    size_t elements = this->GetTotalElements();
    LaunchScalarKernelInplace(
        this->pdata.get() + this->offset,
        elements,
        value,
        defaultBlockSize,
        ScalarOp::MUL
    );
}

template<typename T>
void CudaTensor<T>::Div_(T value)
{
    if (value == T(0))
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
    }

    size_t elements = this->GetTotalElements();
    LaunchScalarKernelInplace(
        this->pdata.get() + this->offset,
        elements,
        value,
        defaultBlockSize,
        ScalarOp::DIV
    );
}

// 内存管理
template<typename T>
std::shared_ptr<T> CudaTensor<T>::Allocate(size_t elements) const
{
    auto ptr = AllocateKernel(elements * sizeof(T));
    return shared_ptr<T>((T*)ptr, CudaFreeDirect);
}

// 张量操作
template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Contiguous() const
{
    auto result = CreateCudaTensor(this->shape);
    if (this->IsContiguous())
    {
        CopyDataContiguous(result->pdata.get());
    }
    else
    {
        CopyDataNoncontiguous(result->pdata.get());
    }
    return result;
}
template<typename T>
void CudaTensor<T>::Contiguous_()
{
    if (this->IsContiguous())
        return;
    auto newPtr = Allocate(this->GetTotalElements());
    CopyDataNoncontiguous(newPtr.get());
    this->pdata = newPtr;
    this->offset = 0;
    this->strides = this->GetContiguousStrides();
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

template<typename T>
shared_ptr<T> CudaTensor<T>::CopyDataToHost() const
{
    size_t elements = this->GetTotalElements();
    size_t totalBytes = elements * sizeof(T);
    T* hostData = new T[elements];

    if (this->IsContiguous())
    {
        CUDA_CHECK(cudaMemcpy(hostData, this->pdata.get() + this->offset, totalBytes, cudaMemcpyDeviceToHost));
    }
    else
    {
        auto tensor = Contiguous();
        CUDA_CHECK(cudaMemcpy(hostData, tensor->GetData().get() + tensor->GetOffset(), totalBytes, cudaMemcpyDeviceToHost));
    }
    return shared_ptr<T>(hostData, [](T* p) { delete[] p; });
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Create() const
{
    return make_shared<CudaTensor<T>>();
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Create(const std::vector<int>& shape) const
{
    return make_shared<CudaTensor<T>>(shape);
}

template<typename T>
bool CudaTensor<T>::HasZero() const
{
    size_t elements = this->GetTotalElements();
    return LaunchHasZeroKernel(this->pdata.get() + this->offset, elements, defaultBlockSize);
}

template<typename T>
void CudaTensor<T>::RepeatRecursive(int dim, const std::vector<int>& repeats, T*& destData, vector<int>& indices) const
{
    throw std::runtime_error("RepeatRecursive for CUDA tensor not implemented. Use Contiguous() first.");
}

// 保护方法实现
template<typename T>
void CudaTensor<T>::CopyDataContiguous(T* dest) const
{
    size_t elements = this->GetTotalElements();
    size_t totalBytes = elements * sizeof(T);
    CUDA_CHECK(cudaMemcpy(dest, this->pdata.get() + this->offset, totalBytes, cudaMemcpyDeviceToHost));
}

template<typename T>
void CudaTensor<T>::CopyDataNoncontiguous(T* dest) const
{
    auto elements = this->GetTotalElements();
    int dims = shape.size();
    auto dShape = CopyShapeToDevice();
    auto dStrides = CopyStridesToDevice();
    auto gridSize = GetDefaultGridSize(elements);
    CopyFromNoncontiguousKernel<<<gridSize, defaultBlockSize>>>(this->pdata.get(), newPtr.get(), dShape, dStrides, dims, elements);
}

template<typename T>
void CudaTensor<T>::CopyRecursive(int dim, T*& dest, size_t currOffset) const
{
    // 简化实现，实际应使用 CUDA 核函数
    if (dim == this->shape.size() - 1)
    {
        T* currData = this->pdata.get() + this->offset + currOffset;
        for (auto i = 0; i < this->shape[dim]; ++i)
        {
            T temp;
            CUDA_CHECK(cudaMemcpy(&temp, currData, sizeof(T), cudaMemcpyDeviceToHost));
            *dest++ = temp;
            currData += this->strides[dim];
        }
        return;
    }
    for (int i = 0; i < this->shape[dim]; ++i)
    {
        size_t newOffset = currOffset + i * this->strides[dim];
        CopyRecursive(dim + 1, dest, newOffset);
    }
}

template<typename T>
void CudaTensor<T>::FillValue(T value)
{
    size_t elements = this->GetTotalElements();
    LaunchFillKernel(this->pdata.get() + this->offset, elements, value, defaultBlockSize);
}

template<typename T>
void CudaTensor<T>::FillValueWithStep(T start, T step)
{
    size_t elements = this->GetTotalElements();
    LaunchArrangeKernel(this->pdata.get() + this->offset, elements, start, step, defaultBlockSize);
}

template<typename T>
int CudaTensor<T>::GetDefaultGridSize(size_t elements) const
{
    return (elements + defaultBlockSize - 1) / defaultBlockSize;
}

template<typename T>
void CudaTensor<T>::CheckAndBroadcast(const Tensor<T>& other, std::vector<int>& broadcastShape) const
{
    if (this->IsBroadcastable(this->shape, other.GetShape()))
    {
        broadcastShape = this->GetBroadcastShape(this->shape, other.GetShape());
    }
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::BroadcastOp(const Tensor<T>& other, const std::string& op) const
{
    std::vector<int> broadcastShape = this->GetBroadcastShape(this->shape, other.GetShape());
    auto result = CreateCudaTensor(broadcastShape);

    // 简化实现，实际应使用广播核函数
    throw std::runtime_error("Broadcast operation not yet implemented for CUDA tensor");
}

// 工厂方法
template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::CreateCudaTensor()
{
    return make_shared<CudaTensor<T>>();
}

template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::CreateCudaTensor(const std::vector<int>& shape)
{
    return make_shared<CudaTensor<T>>(shape);
}

template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::Full(const std::vector<int>& shape, T value)
{
    auto tensor = CreateCudaTensor(shape);
    tensor->FillValue(value);
    return tensor;
}

template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::Ones(const std::vector<int>& shape)
{
    return Full(shape, T(1));
}

template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::Zeros(const std::vector<int>& shape)
{
    auto tensor = CreateCudaTensor(shape);
    size_t totalBytes = tensor->GetTotalElements() * sizeof(T);
    CUDA_CHECK(cudaMemset(tensor->pdata.get(), 0, totalBytes));
    return tensor;
}

template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::Arrange(const std::vector<int>& shape, T start, T step)
{
    auto tensor = CreateCudaTensor(shape);
    tensor->FillValueWithStep(start, step);
    return tensor;
}

template<typename T>
std::shared_ptr<char> CudaTensor<T>::CopyShapeToDevice() const
{
    auto dShape = Allocate(shape.size());
    CUDA_CHECK(cudaMemcpyAsync(dShape.get(), shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice), nullptr);
    return dShape;
}
template<typename T>
std::shared_ptr<char> CudaTensor<T>::CopyStridesToDevice() const
{
    auto dStrides = Allocate(strides.size() * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(dStrides.get(), strides.data(), strides.size() * sizeof(int), cudaMemcpyHostToDevice), nullptr);
    return dStrides;
}

template class CudaTensor<float>;
template class CudaTensor<double>;
template class CudaTensor<int>;

