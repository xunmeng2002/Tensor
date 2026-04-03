#pragma once
#include "Tensor.h"
#include <functional>
#include <memory>


template<typename T>
class CpuTensor : public Tensor<T>
{
public:
    using ValueType = T;
    CpuTensor() noexcept :Tensor<T>(DeviceType::CPU) {}
    CpuTensor(const std::vector<int>& shape) noexcept
        :Tensor<T>(shape, DeviceType::CPU)
    {
		this->pdata = Allocate(this->GetTotalElements());
    }
    CpuTensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape) noexcept
        :Tensor<T>(data, deleter, offset, shape, DeviceType::CPU)
    {}
    CpuTensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape) noexcept
        :Tensor<T>(data, offset, shape, DeviceType::CPU)
    {}
    virtual ~CpuTensor() = default;

    virtual shared_ptr<Tensor<T>> Add(const Tensor<T>& other) const override
    {
        if (!this->CheckDevice(other) || !this->CheckShape(other))
        {
            return nullptr;
        }
        auto result = this->CreateCpuTensor(this->shape);
        auto resultIt = result->begin();
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther, ++resultIt)
        {
            *resultIt = *it1 + *itOther;
        }
        return result;
    }
    virtual shared_ptr<Tensor<T>> Sub(const Tensor<T>& other) const override
    {
        if (!this->CheckDevice(other) || !this->CheckShape(other))
        {
            return nullptr;
        }
        auto result = this->CreateCpuTensor(this->shape);
        auto resultIt = result->begin();
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther, ++resultIt)
        {
            *resultIt = *it1 - *itOther;
        }
        return result;
    }
    virtual shared_ptr<Tensor<T>> Mul(const Tensor<T>& other) const override
    {
        if (!this->CheckDevice(other) || !this->CheckShape(other))
        {
            return nullptr;
        }
        auto result = this->CreateCpuTensor(this->shape);
        auto resultIt = result->begin();
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther, ++resultIt)
        {
            *resultIt = (*it1) * (*itOther);
        }
        return result;
    }
    virtual shared_ptr<Tensor<T>> Div(const Tensor<T>& other) const override
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
        auto resultIt = result->begin();
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther, ++resultIt)
        {
            *resultIt = (*it1) / (*itOther);
        }
        return result;
    }
    virtual shared_ptr<Tensor<T>> Add(T value) const override
    {
        auto result = this->CreateCpuTensor(this->shape);
        auto resultIt = result->begin();
        for (auto it = begin(); it != end(); ++it, ++resultIt)
        {
            *resultIt = *it + value;
        }
        return result;
    }
    virtual shared_ptr<Tensor<T>> Sub(T value) const override
    {
        auto result = this->CreateCpuTensor(this->shape);
        auto resultIt = result->begin();
        for (auto it = begin(); it != end(); ++it, ++resultIt)
        {
            *resultIt = *it - value;
        }
        return result;
    }
    virtual shared_ptr<Tensor<T>> Mul(T value) const override
    {
        auto result = this->CreateCpuTensor(this->shape);
        auto resultIt = result->begin();
        for (auto it = begin(); it != end(); ++it, ++resultIt)
        {
            *resultIt = *it * value;
        }
        return result;
    }
    virtual shared_ptr<Tensor<T>> Div(T value) const override
    {
        if (value == T(0))
        {
            throw std::runtime_error("Division by zero in tensor element-wise division");
        }
        auto result = this->CreateCpuTensor(this->shape);
        auto resultIt = result->begin();
        for (auto it = begin(); it != end(); ++it, ++resultIt)
        {
            *resultIt = *it / value;
        }
        return result;
    }
    virtual void Add_(const Tensor<T>& other) override
    {
        if (!this->CheckDevice(other) || !this->CheckShape(other))
        {
            return;
        }
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther)
        {
            *it1 = *it1 + *itOther;
        }
    }
    virtual void Sub_(const Tensor<T>& other) override
    {
        if (!this->CheckDevice(other) || !this->CheckShape(other))
        {
            return;
        }
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther)
        {
            *it1 = *it1 - *itOther;
        }
    }
    virtual void Mul_(const Tensor<T>& other) override
    {
        if (!this->CheckDevice(other) || !this->CheckShape(other))
        {
            return;
        }
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther)
        {
            *it1 = (*it1) * (*itOther);
        }
    }
    virtual void Div_(const Tensor<T>& other) override
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
        auto itOther = static_cast<const CpuTensor<T>&>(other).begin();
        for (auto it1 = begin(); it1 != end(); ++it1, ++itOther)
        {
            *it1 = (*it1) / (*itOther);
        }
    }
    virtual void Add_(T value) override
    {
        for (auto it = begin(); it != end(); ++it)
        {
            *it = *it + value;
        }
    }
    virtual void Sub_(T value) override
    {
        for (auto it = begin(); it != end(); ++it)
        {
            *it = *it - value;
        }
    }
    virtual void Mul_(T value) override
    {
        for (auto it = begin(); it != end(); ++it)
        {
            *it = *it * value;
        }
    }
    virtual void Div_(T value) override
    {
        if (value == T(0))
        {
            throw std::runtime_error("Division by zero in tensor element-wise division");
            return;
        }
        for (auto it = begin(); it != end(); ++it)
        {
            *it = *it / value;
        }
    }

    class iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        iterator() : tensor(nullptr), linearIndex(0) {}
        iterator(CpuTensor<T>* tensor, size_t linear_idx)
            : tensor(tensor), linearIndex(linear_idx)
        {
        }
        reference operator*() const
        {
            if (!tensor || linearIndex >= tensor->GetTotalElements())
            {
                throw std::out_of_range("Iterator out of range");
            }
            std::vector<int> indices = tensor->LinearToIndices(linearIndex);
            size_t offset = tensor->CalculateOffset(indices);
            return *(tensor->pdata.get() + offset);
        }
        pointer operator->() const
        {
            return &(**this);
        }
        iterator& operator++()
        {
            ++linearIndex;
            return *this;
        }
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator==(const iterator& other) const
        {
            return tensor == other.tensor && linearIndex == other.linearIndex;
        }
        bool operator!=(const iterator& other) const
        {
            return !(*this == other);
        }
    private:
        CpuTensor<T>* tensor;
        size_t linearIndex;
    };
    class const_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator() : tensor(nullptr), linearIndex(0) {}
        const_iterator(const CpuTensor<T>* tensor, size_t linear_idx)
            : tensor(tensor), linearIndex(linear_idx)
        {
        }
        reference operator*() const
        {
            if (!tensor || linearIndex >= tensor->GetTotalElements())
            {
                throw std::out_of_range("Iterator out of range");
            }
            std::vector<int> indices = tensor->LinearToIndices(linearIndex);
            size_t currOffset = tensor->CalculateOffset(indices);
            return *(tensor->pdata.get() + currOffset);
        }
        pointer operator->() const
        {
            return &(**this);
        }
        const_iterator& operator++()
        {
            ++linearIndex;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator==(const const_iterator& other) const
        {
            return tensor == other.tensor && linearIndex == other.linearIndex;
        }
        bool operator!=(const const_iterator& other) const
        {
            return !(*this == other);
        }
    private:
        const CpuTensor<T>* tensor;
        size_t linearIndex;
    };

    iterator begin()
    {
        return iterator(this, 0);
    }
    const_iterator begin() const
    {
        return const_iterator(this, 0);
    }
    const_iterator cbegin() const
    {
        return const_iterator(this, 0);
    }
    iterator end()
    {
        return iterator(this, this->GetTotalElements());
    }
    const_iterator end() const
    {
        return const_iterator(this, this->GetTotalElements());
    }
    const_iterator cend() const
    {
        return const_iterator(this, this->GetTotalElements());
    }

    virtual std::shared_ptr<T> Allocate(size_t elements) const override
    {
        return shared_ptr<T>(new T[elements * sizeof(T)], [](T* p) {delete[] p; });
    }
    
    virtual const T& At(const std::vector<int>& indices, bool checkBounds = true) const override
    {
        size_t currOffset = checkBounds ? this->CalculateOffset(indices) : this->CalculateOffsetUnChecked(indices);
        return *(this->pdata.get() + currOffset);
    }
    virtual T& At(const std::vector<int>& indices, bool checkBounds = true) override
    {
        size_t currOffset = checkBounds ? this->CalculateOffset(indices) : this->CalculateOffsetUnChecked(indices);
        return *(this->pdata.get() + currOffset);
    }
    
    virtual shared_ptr<Tensor<T>> Contiguous() const override
    {
        auto result = make_shared<CpuTensor>();
        if (this->IsContiguous())
        {
            *result = *this;
            return result;
        }
        result->pdata = Allocate(this->GetTotalElements());
        result->offset = 0;
        result->shape = this->shape;
        result->strides = result->GetContiguousStrides();
        std::vector<int> indices(result->shape.size(), 0);
        auto destPtr = result->pdata.get();
        this->CopyRecursive(0, destPtr, 0);
        return result;
    }
    virtual void Contiguous_() override
    {
        if (this->IsContiguous()) return;

        auto newPtr = Allocate(this->GetTotalElements());
        auto destPtr = newPtr.get();
        this->CopyRecursive(0, destPtr, 0);
        this->pdata = newPtr;
        this->offset = 0;
        this->strides = this->GetContiguousStrides();
    }
    
    virtual shared_ptr<Tensor<T>> Create() const override
    {
        return make_shared<CpuTensor<T>>();
    }
    virtual shared_ptr<Tensor<T>> Create(const std::vector<int>& shape) const override
    {
        return make_shared<CpuTensor<T>>(shape);
    }
    static shared_ptr<CpuTensor<T>> CreateCpuTensor()
    {
        return make_shared<CpuTensor<T>>();
    }
    static shared_ptr<CpuTensor<T>> CreateCpuTensor(const std::vector<int>& shape)
    {
        return make_shared<CpuTensor<T>>(shape);
    }
    
    static void DeleteNone(void* ptr)
    {
    }
    static void DeleteDirect(void* ptr)
    {
        delete[] ptr;
    }
    static void DeletePool(void* ptr)
    {
    }

    virtual void CopyDataContiguous(T* dest) const override
    {
        auto elements = this->GetTotalElements();
        memcpy(dest, this->pdata.get() + this->offset, elements * sizeof(T));
    }
    virtual void CopyDataNoncontiguous(T* dest) const override
    {
        CopyRecursive(0, dest, 0);
    }
    virtual shared_ptr<T> CopyDataToHost() const override
    {
		return this->pdata;
	}
    virtual bool HasZero() const override
    {
        for (auto it = begin(); it != end(); ++it)
        {
            if (*it == T(0))
            {
                return true;
            }
        }
        return false;
    }
	virtual void RepeatRecursive(int dim, const std::vector<int>& repeats, T*& destData, vector<int>& indices) const override
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
                RepeatRecursive(dim + 1, repeats, destData, indices);
            }
        }
    }


    static shared_ptr<CpuTensor<T>> Arrange(const std::vector<int>& shape, T start, T step)
    {
        auto tensor = CreateCpuTensor(shape);
        tensor->FillValueWithStep(start, step);
        return tensor;
    }
    static shared_ptr<CpuTensor<T>> Full(const std::vector<int>& shape, T value)
    {
        auto tensor = CreateCpuTensor(shape);
        tensor->FillValue(value);
        return tensor;
    }
    static shared_ptr<CpuTensor<T>> Ones(const std::vector<int>& shape)
    {
        auto tensor = CreateCpuTensor(shape);
        tensor->FillValue(T(1));
        return tensor;
    }
    static shared_ptr<CpuTensor<T>> Zeros(const std::vector<int>& shape)
    {
        auto tensor = CreateCpuTensor(shape);
        T* data_ptr = tensor->pdata.get();
        size_t total_bytes = tensor->GetTotalElements() * sizeof(T);
        std::memset(data_ptr, 0, total_bytes);
        return tensor;
    }

protected:
    void CopyRecursive(int dim, T*& dest, size_t currOffset) const
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
            CopyRecursive(dim + 1, dest, newOffset);
        }
    }
    void FillValue(T value)
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
    void FillValueWithStep(T start, T step)
    {
        size_t elements = this->GetTotalElements();
        auto dataPtr = this->pdata.get() + this->offset_bytes;
        for (auto i = 0; i < elements; ++i)
        {
            dataPtr[i] = start + i * step;
        }
    }
};

