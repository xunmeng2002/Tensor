#pragma once
#include "CudaTensor.h"
#include "CudaTensorKernel.cuh"



// 算术运算实现
template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Add(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return nullptr;
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
        return nullptr;
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
        return nullptr;
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
        return nullptr;
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
std::shared_ptr<T> CudaTensor<T>::Allocate(size_t elements)
{
    auto ptr = AllocateKernel(elements * sizeof(T));
    return shared_ptr<T>((T*)ptr, CudaFreeDirect);
}
template<typename T>
std::shared_ptr<uint8_t> CudaTensor<T>::AllocateBytes(size_t bytes)
{
	auto ptr = AllocateKernel(bytes);
	return shared_ptr<uint8_t>((uint8_t*)ptr, CudaFreeDirect);
}

template<typename T>
T CudaTensor<T>::At(const std::vector<int>& indices, bool checkBounds) const
{
    size_t currOffset = checkBounds ? this->CalculateOffset(indices) : this->CalculateOffsetUnChecked(indices);
	auto destData = this->pdata.get() + currOffset;
    T hostValue = 0;
    CUDA_CHECK(cudaMemcpy(&hostValue, destData, sizeof(T), cudaMemcpyDeviceToHost));
	return hostValue;
}
template<typename T>
T& CudaTensor<T>::At(const std::vector<int>& indices, bool checkBounds)
{
	throw std::runtime_error("Direct element access is not supported for CudaTensor. Use CopyDataToHost() to access data on host.");
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
bool CudaTensor<T>::HasZero() const
{
    size_t elements = this->GetTotalElements();
    return LaunchHasZeroKernel(this->pdata.get() + this->offset, elements, defaultBlockSize);
}
template<typename T>
bool CudaTensor<T>::CheckEqual(const vector<T>& data) const
{
    size_t elements = this->GetTotalElements();
    auto dData = CopyVectorToDevice(data);
    auto dShape = CopyVectorToDevice(this->shape);
    auto dStrides = CopyVectorToDevice(this->strides);

    bool* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_result, 1, sizeof(bool)));
    int gridSize = (elements + defaultBlockSize - 1) / defaultBlockSize;

    CheckEqualKernel<T><<<gridSize, defaultBlockSize>>>(this->pdata.get(), (T*)dData.get(), (int*)dShape.get(), (int*)dStrides.get(),
        this->shape.size(), this->offset, elements, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    bool result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}


// 工厂方法
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
    int dims = this->shape.size();
    auto dShape = CopyVectorToDevice(this->shape);
    auto dStrides = CopyVectorToDevice(this->strides);
    auto gridSize = GetDefaultGridSize(elements);
    CopyFromNoncontiguousKernel<<<gridSize, defaultBlockSize>>>(this->pdata.get(), dest, (int*)dShape.get(), (int*)dStrides.get(), dims, elements);
}
template<typename T>
void CudaTensor<T>::CopyDataRepeat(Tensor<T>* repeatTensor, const std::vector<int>& repeats) const
{
    auto elements = repeatTensor->GetTotalElements();
    int dims = this->shape.size();
    auto dShape = CopyVectorToDevice(this->shape);
    auto dStrides = CopyVectorToDevice(this->strides);
    auto dDestShape = CopyVectorToDevice(repeatTensor->GetShape());
    auto gridSize = GetDefaultGridSize(elements);
    RepeatKernel<<<gridSize, defaultBlockSize>>>(this->pdata.get(), repeatTensor->GetData().get(), (int*)dShape.get(), (int*)dStrides.get(), (int*)dDestShape.get(), this->offset, dims, elements);
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
template<typename VecT>
std::shared_ptr<uint8_t> CudaTensor<T>::CopyVectorToDevice(const vector<VecT>& values)
{
    auto dValues = AllocateBytes(values.size() * sizeof(VecT));
    CUDA_CHECK(cudaMemcpyAsync(dValues.get(), values.data(), values.size() * sizeof(VecT), cudaMemcpyHostToDevice, nullptr));
    return dValues;
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




template class CudaTensor<float>;
template class CudaTensor<double>;
template class CudaTensor<int8_t>;
template class CudaTensor<uint8_t>;
template class CudaTensor<int16_t>;
template class CudaTensor<uint16_t>;
template class CudaTensor<int32_t>;
template class CudaTensor<uint32_t>;
template class CudaTensor<int64_t>;
template class CudaTensor<uint64_t>;

