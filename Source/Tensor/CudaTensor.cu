#pragma once
#include "CudaTensor.h"
#include "CudaTensorKernel.cuh"

// 辅助函数实现
template<typename T>
void CopyDataToHostInternal(const CudaTensor<T>& tensor, T* hostData)
{
    // 非连续拷贝的辅助实现
    std::vector<int> indices(tensor.GetShape().size(), 0);
    size_t idx = 0;
    // 遍历所有元素（简化实现，实际应使用更高效的方法）
    // 这里只是框架，完整实现需要递归遍历
}


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

// 元素访问
template<typename T>
const T& CudaTensor<T>::At(const std::vector<int>& indices, bool checkBounds) const
{
    throw std::runtime_error("Direct access to CUDA tensor data not supported. Use CopyDataToHost() first.");
}

template<typename T>
T& CudaTensor<T>::At(const std::vector<int>& indices, bool checkBounds)
{
    throw std::runtime_error("Direct access to CUDA tensor data not supported. Use CopyDataToHost() first.");
}

// 张量操作
template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Contiguous() const
{
    auto result = CreateCudaTensor();
    if (this->IsContiguous())
    {
        *result = *this;
        return result;
    }

    result->GetData() = Allocate(this->GetTotalElements());
    result->offset = 0;
    result->shape = this->shape;
    result->strides = result->GetContiguousStrides();

    // 使用 CUDA 内存拷贝
    size_t totalBytes = this->GetTotalElements() * sizeof(T);

    // 如果不连续，需要逐元素拷贝（这里简化处理）
    // 实际实现中应该使用更高效的方法
    T* hostBuffer = new T[this->GetTotalElements()];
    CopyDataToHostInternal(hostBuffer);

    CUDA_CHECK(cudaMemcpy(result->GetData().get(), hostBuffer, totalBytes, cudaMemcpyHostToDevice));

    delete[] hostBuffer;
    return result;
}

template<typename T>
void CudaTensor<T>::Contiguous_()
{
    if (this->IsContiguous()) return;

    auto newPtr = Allocate(this->GetTotalElements());

    // 拷贝数据到新内存
    T* hostBuffer = new T[this->GetTotalElements()];
    CopyDataToHostInternal(hostBuffer);

    size_t totalBytes = this->GetTotalElements() * sizeof(T);
    CUDA_CHECK(cudaMemcpy(newPtr.get(), hostBuffer, totalBytes, cudaMemcpyHostToDevice));

    delete[] hostBuffer;

    this->pdata = newPtr;
    this->offset = 0;
    this->strides = this->GetContiguousStrides();
}

template<typename T>
shared_ptr<T> CudaTensor<T>::CopyDataToHost() const
{
    size_t elements = this->GetTotalElements();
    T* hostData = new T[elements];

    if (this->IsContiguous())
    {
        size_t totalBytes = elements * sizeof(T);
        CUDA_CHECK(cudaMemcpy(hostData, this->pdata.get() + this->offset, totalBytes, cudaMemcpyDeviceToHost));
    }
    else
    {
        CopyDataToHostInternal(hostData);
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
    CopyDataToHostInternal(dest);
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




template class CudaTensor<float>;
template class CudaTensor<double>;
template class CudaTensor<int>;

