#include "CpuTensor.h"


template<typename T>
CpuTensor<T>::CpuTensor() noexcept
    :Tensor<T>(DeviceType::CPU)
{}
template<typename T>
CpuTensor<T>::CpuTensor(const std::vector<int>& shape) noexcept
    :Tensor<T>(shape, DeviceType::CPU)
{
    this->pdata = Allocate(this->GetTotalElements());
}
template<typename T>
CpuTensor<T>::CpuTensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape) noexcept 
    :Tensor<T>(data, deleter, offset, shape, DeviceType::CPU)
{
}
template<typename T>
CpuTensor<T>::CpuTensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape) noexcept
    :Tensor<T>(data, offset, shape, DeviceType::CPU)
{
}

template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Add(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return nullptr;
    }
    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] + other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) + other.At(indices);
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Sub(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return nullptr;
    }
    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] - other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) - other.At(indices);
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Mul(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return nullptr;
    }
    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] * other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) * other.At(indices);
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Div(const Tensor<T>& other) const
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return nullptr;
    }
    if (other.HasZero())
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
    }

    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] / other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) / other.At(indices);
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Add(T value) const
{
    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] + value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) + value;
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Sub(T value) const
{
    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] - value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) - value;
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Mul(T value) const
{
    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] * value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) * value;
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Div(T value) const
{
    if (value == T(0))
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
    }
    auto result = this->CreateCpuTensor(this->shape);
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = (this->pdata.get())[i] / value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            result->GetData().get()[i] = At(indices) / value;
            this->UpdateIndices(indices);
        }
    }
    return result;
}
template<typename T>
void CpuTensor<T>::Add_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return;
    }
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] += other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) +=  other.At(indices);
            this->UpdateIndices(indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::Sub_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return;
    }
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] -= other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) -= other.At(indices);
            this->UpdateIndices(indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::Mul_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return;
    }
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] *= other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) *= other.At(indices);
            this->UpdateIndices(indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::Div_(const Tensor<T>& other)
{
    if (!this->CheckDevice(other) || !this->CheckShape(other))
    {
        return;
    }
    if (other.HasZero())
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
        return;
    }
    auto elements = this->GetTotalElements();
    if (this->IsContiguous() && other.IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] /= other.GetData().get()[i];
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) /= other.At(indices);
            this->UpdateIndices(indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::Add_(T value)
{
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] += value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) += value;
            this->UpdateIndices(indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::Sub_(T value)
{
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] -= value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) -= value;
            this->UpdateIndices(indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::Mul_(T value)
{
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] /= value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) *= value;
            this->UpdateIndices(indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::Div_(T value)
{
    if (value == T(0))
    {
        throw std::runtime_error("Division by zero in tensor element-wise division");
        return;
    }
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        for (uint64_t i = 0; i < elements; ++i)
        {
            (this->pdata.get())[i] /= value;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            At(indices) /= value;
            this->UpdateIndices(indices);
        }
    }
}

template<typename T>
std::shared_ptr<T> CpuTensor<T>::Allocate(size_t elements)
{
    return shared_ptr<T>(new T[elements * sizeof(T)], [](T* ptr) {delete[] ptr; });
}
template<typename T>
T CpuTensor<T>::At(const std::vector<int>& indices, bool checkBounds) const
{
    size_t currOffset = checkBounds ? this->CalculateOffset(indices) : this->CalculateOffsetUnChecked(indices);
    return *(this->pdata.get() + currOffset);
}
template<typename T>
T& CpuTensor<T>::At(const std::vector<int>& indices, bool checkBounds)
{
    size_t currOffset = checkBounds ? this->CalculateOffset(indices) : this->CalculateOffsetUnChecked(indices);
    return *(this->pdata.get() + currOffset);
}
template<typename T>
bool CpuTensor<T>::CheckEqual(const vector<T>& data) const
{
    size_t elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        auto ptr = this->pdata.get() + this->offset;
        for (size_t i = 0; i < elements; ++i)
        {
            if (*ptr++ != data[i])
                return false;
        }
    }
    else
    {
        int dims = (int)this->shape.size();
        vector<int> indices(dims);
        for (size_t i = 0; i < elements; ++i)
        {
            T value = this->At(indices, false);
            if (value != data[i])
                return false;
            this->UpdateIndices(indices);
        }
    }
    return true;
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Contiguous() const
{
    auto result = CreateCpuTensor(this->shape);
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
void CpuTensor<T>::Contiguous_()
{
    if (this->IsContiguous())
        return;

    auto newPtr = Allocate(this->GetTotalElements());
    auto destPtr = newPtr.get();
    CopyDataNoncontiguous(destPtr);
    this->pdata = newPtr;
    this->offset = 0;
    this->strides = this->GetContiguousStrides();
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Create() const
{
    return make_shared<CpuTensor<T>>();
}
template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Create(const std::vector<int>& shape) const
{
    return make_shared<CpuTensor<T>>(shape);
}
template<typename T>
shared_ptr<CpuTensor<T>> CpuTensor<T>::CreateCpuTensor()
{
    return make_shared<CpuTensor<T>>();
}
template<typename T>
shared_ptr<CpuTensor<T>> CpuTensor<T>::CreateCpuTensor(const std::vector<int>& shape)
{
    return make_shared<CpuTensor<T>>(shape);
}
template<typename T>
shared_ptr<CpuTensor<T>> CpuTensor<T>::Full(const std::vector<int>& shape, T value)
{
    auto tensor = CreateCpuTensor(shape);
    tensor->FillValue(value);
    return tensor;
}
template<typename T>
shared_ptr<CpuTensor<T>> CpuTensor<T>::Ones(const std::vector<int>& shape)
{
    auto tensor = CreateCpuTensor(shape);
    tensor->FillValue(T(1));
    return tensor;
}
template<typename T>
shared_ptr<CpuTensor<T>> CpuTensor<T>::Zeros(const std::vector<int>& shape)
{
    auto tensor = CreateCpuTensor(shape);
    T* data_ptr = tensor->pdata.get();
    size_t total_bytes = tensor->GetTotalElements() * sizeof(T);
    std::memset(data_ptr, 0, total_bytes);
    return tensor;
}
template<typename T>
shared_ptr<CpuTensor<T>> CpuTensor<T>::Arrange(const std::vector<int>& shape, T start, T step)
{
    auto tensor = CreateCpuTensor(shape);
    tensor->FillValueWithStep(start, step);
    return tensor;
}
template<typename T>
bool CpuTensor<T>::HasZero() const
{
    auto elements = this->GetTotalElements();
    if (this->IsContiguous())
    {
        auto ptr = this->pdata.get() + this->offset;
        for (size_t i = 0; i < elements; ++i)
        {
            if (*ptr++ == T(0))
                return true;
        }
    }
    else
    {
        vector<int> indices(this->shape.size());
        for (size_t i = 0; i < elements; ++i)
        {
            if (At(indices) == T(0))
                return true;
            this->UpdateIndices(indices);
        }
    }
    return false;
}


template<typename T>
void CpuTensor<T>::CopyDataContiguous(T* dest) const
{
    auto elements = this->GetTotalElements();
    memcpy(dest, this->pdata.get() + this->offset, elements * sizeof(T));
}
template<typename T>
void CpuTensor<T>::CopyDataNoncontiguous(T* dest) const
{
    CopyDataRecursive(0, dest, 0);
}
template<typename T>
void CpuTensor<T>::CopyDataRepeat(Tensor<T>* repeatTensor, const std::vector<int>& repeats) const
{
    auto destPtr = repeatTensor->GetData().get();
    std::vector<int> indices(this->shape.size(), 0);
    this->CopyDataRepeatRecursive(0, repeats, destPtr, indices);
}
template<typename T>
shared_ptr<T> CpuTensor<T>::CopyDataToHost() const
{
    return this->pdata;
}

template<typename T>
void CpuTensor<T>::CopyDataRecursive(int dim, T*& dest, size_t currOffset) const
{
    if (dim == this->shape.size() - 1)
    {
        T* currData = this->pdata.get() + this->offset + currOffset;
        for (auto i = 0; i < this->shape[dim]; ++i)
        {
            *dest++ = *currData;
            currData += this->strides[dim];
        }
        return;
    }
    for (int i = 0; i < this->shape[dim]; ++i)
    {
        size_t newOffset = currOffset + i * this->strides[dim];
        CopyDataRecursive(dim + 1, dest, newOffset);
    }
}
template<typename T>
void CpuTensor<T>::CopyDataRepeatRecursive(int dim, const std::vector<int>& repeats, T*& destData, vector<int>& indices) const
{
    if (dim == this->shape.size() - 1)
    {
        size_t curr_offset = this->offset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            curr_offset += indices[i] * this->strides[i];
        }
        auto dataStart = this->pdata.get() + curr_offset;
        for (auto j = 0; j < this->shape[dim]; ++j)
        {
            *destData++ = *(dataStart + j * this->strides[dim]);
        }
        return;
    }
    for (int i = 0; i < repeats[dim]; ++i)
    {
        for (int j = 0; j < this->shape[dim]; ++j)
        {
            indices[dim] = j;
            CopyDataRepeatRecursive(dim + 1, repeats, destData, indices);
        }
    }
}
template<typename T>
void CpuTensor<T>::FillValue(T value)
{
    size_t elements = this->GetTotalElements();
    size_t sizes = elements * sizeof(T);
    if constexpr (std::is_same_v<T, char> || std::is_same_v<T, unsigned char> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>)
    {
        memset(this->pdata.get() + this->offset, value, sizes);
    }
    else if (value == T(0))
    {
        memset(this->pdata.get() + this->offset, 0, sizes);
    }
    else
    {
        T* ptr = this->pdata.get() + this->offset;
        std::fill(ptr, ptr + elements, value);
    }
}
template<typename T>
void CpuTensor<T>::FillValueWithStep(T start, T step)
{
    size_t elements = this->GetTotalElements();
    auto dataPtr = this->pdata.get() + this->offset;
    for (auto i = 0; i < elements; ++i)
    {
        dataPtr[i] = start + (T)i * step;
    }
}



template class CpuTensor<float>;
template class CpuTensor<double>;
template class CpuTensor<int8_t>;
template class CpuTensor<uint8_t>;
template class CpuTensor<int16_t>;
template class CpuTensor<uint16_t>;
template class CpuTensor<int32_t>;
template class CpuTensor<uint32_t>;
template class CpuTensor<int64_t>;
template class CpuTensor<uint64_t>;


