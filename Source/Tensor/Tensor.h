#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <memory>
#include <numeric>

using namespace std;

enum class DeviceType : int { CPU, CUDA };
// 运算类型枚举
enum class ElementwiseOp { ADD, SUB, MUL, DIV };
enum class ScalarOp { ADD, SUB, MUL, DIV };

template<typename T>
class Tensor
{
public:
    using ValueType = T;
    Tensor(DeviceType device) noexcept :pdata(nullptr), offset(0), device(device){}
    Tensor(const vector<int>& shape, DeviceType device) noexcept :shape(shape), offset(0), device(device)
    {
        this->strides = GetContiguousStrides(shape);
    }
    Tensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape, DeviceType device) noexcept
        :pdata(data, deleter), offset(offset), shape(shape), device(device)
    {
        strides = GetContiguousStrides(shape);
    }
    Tensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, DeviceType device) noexcept
        :pdata(data), offset(offset), shape(shape), device(device)
    {
        strides = GetContiguousStrides(shape);
    }
    virtual ~Tensor() = default;

    Tensor(const Tensor& other) noexcept
    {
        pdata = other.pdata;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;
    }
    Tensor& operator=(const Tensor& other) noexcept
    {
        pdata = other.pdata;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;
        return *this;
    }
    Tensor(Tensor&& other) noexcept
    {
        pdata = std::move(other.pdata);
        offset = other.offset;
        shape = std::move(other.shape);
		strides = std::move(other.strides);
    }
    Tensor& operator=(Tensor&& other) noexcept
    {
        if (this != &other)
        {
            pdata = std::move(other.pdata);
            offset = other.offset;
            shape = std::move(other.shape);
            strides = std::move(other.strides);
        }
        return *this;
    }
    T& operator[](const std::vector<int>& indices)
    {
        size_t offset = CalculateOffset(indices);
        return *(pdata.get() + offset);
    }
    const T& operator[](const std::vector<int>& indices) const
    {
        size_t offset = CalculateOffset(indices);
        return *(pdata.get() + offset);
    }
    T& operator[](int index)
    {
        if (shape.size() != 1)
        {
            throw std::invalid_argument("Scalar indexing only supported for 1D tensors");
        }
        if (index < 0)
        {
            index += shape[0];
        }
        if (index < 0 || index >= shape[0])
        {
            throw std::out_of_range("Index out of bounds");
        }
        int offset = this->offset + index * strides[0];
        return *(pdata.get() + offset);
    }
    const T& operator[](int index) const
    {
        if (shape.size() != 1)
        {
            throw std::invalid_argument("Scalar indexing only supported for 1D tensors");
        }
        if (index < 0)
        {
            index += shape[0];
        }
        if (index < 0 || index >= shape[0])
        {
            throw std::out_of_range("Index out of bounds");
        }
        int offset = this->offset + index * strides[0];
        return *(pdata.get() + offset);
    }
    template<typename... Indices>
    T& operator()(Indices... indices)
    {
        static_assert((std::is_convertible_v<Indices, int> && ...), "All indices must be convertible to int");
        constexpr size_t N = sizeof...(indices);
        if (N != shape.size())
        {
            throw std::invalid_argument("Expected " + std::to_string(shape.size()) + " indices, got " + std::to_string(N));
        }
        size_t dim = 0;
        bool valid = true;
        size_t offset = this->offset;
        ((valid = valid && (static_cast<int>(indices) >= 0 &&
            static_cast<int>(indices) < shape[dim]), offset += static_cast<int>(indices) * strides[dim], dim++), ...);
        if (!valid)
        {
            throw std::out_of_range("Index out of bounds in tensor access");
        }
        return *(pdata.get() + offset);
    }
    
    virtual shared_ptr<Tensor<T>> Add(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Sub(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Mul(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Div(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Add(T value) const = 0;
    virtual shared_ptr<Tensor<T>> Sub(T value) const = 0;
    virtual shared_ptr<Tensor<T>> Mul(T value) const = 0;
    virtual shared_ptr<Tensor<T>> Div(T value) const = 0;
    virtual void Add_(const Tensor<T>& other) = 0;
    virtual void Sub_(const Tensor<T>& other) = 0;
    virtual void Mul_(const Tensor<T>& other) = 0;
    virtual void Div_(const Tensor<T>& other) = 0;
    virtual void Add_(T value) = 0;
    virtual void Sub_(T value) = 0;
    virtual void Mul_(T value) = 0;
    virtual void Div_(T value) = 0;

    shared_ptr<Tensor> operator+(const Tensor& other) const
    {
        return Add(other);
    }
    shared_ptr<Tensor> operator-(const Tensor& other) const
    {
        return Sub(other);
    }
    shared_ptr<Tensor> operator*(const Tensor& other) const
    {
        return Mul(other);
    }
    shared_ptr<Tensor> operator/(const Tensor& other) const
    {
        return Div(other);
    }
    shared_ptr<Tensor> operator+(T value) const
    {
        return Add(value);
    }
    shared_ptr<Tensor> operator-(T value) const
    {
        return Sub(value);
    }
    shared_ptr<Tensor> operator*(T value) const
    {
        return Mul(value);
    }
    shared_ptr<Tensor> operator/(T value) const
    {
        return Div(value);
    }
    Tensor& operator+=(const Tensor& other)
    {
        Add_(other);
        return *this;
    }
    Tensor& operator-=(const Tensor& other)
    {
        Sub_(other);
        return *this;
    }
    Tensor& operator*=(const Tensor& other)
    {
        Mul_(other);
        return *this;
    }
    Tensor& operator/=(const Tensor& other)
    {
        Div_(other);
        return *this;
    }
    Tensor& operator+=(T value)
    {
        Add_(value);
        return *this;
    }
    Tensor& operator-=(T value)
    {
        Sub_(value);
        return *this;
    }
    Tensor& operator*=(T value)
    {
        Mul_(value);
        return *this;
    }
    Tensor& operator/=(T value)
    {
        Div_(value);
        return *this;
    }


    virtual std::shared_ptr<T> Allocate(size_t elements) const = 0;
    virtual const T& At(const std::vector<int>& indices, bool checkBounds = true) const = 0;
    virtual T& At(const std::vector<int>& indices, bool checkBounds = true) = 0;
    bool CheckDims(const vector<int>& dims)
    {
        if (dims.size() != shape.size())
            return false;
        int maxDim = (int)shape.size() - 1;
        vector<bool> dimCounts(shape.size());
        for (auto dim : dims)
        {
            if (dim < 0)
                dim += (int)shape.size();
            if (dim < 0 || dim > maxDim)
            {
                return false;
            }
            else if (dimCounts[dim])
            {
                return false;
            }
            dimCounts[dim] = true;
        }
        return true;
    }
    shared_ptr<Tensor> Clone() const
    {
        if (IsContiguous())
        {
            return CloneContiguous();
        }
        if (HasValidDenseLayout())
        {
            return CloneDenseView();
        }
        return Contiguous();
    }
    virtual shared_ptr<Tensor<T>> Contiguous() const = 0;
    virtual void Contiguous_() = 0;
    virtual shared_ptr<T> CopyDataToHost() const = 0;
    virtual shared_ptr<Tensor<T>> Create() const = 0;
    virtual shared_ptr<Tensor<T>> Create(const std::vector<int>& shape) const = 0;
    
    int Dim() const
    {
        return (int)shape.size();
	}
    shared_ptr<Tensor> Expand(const std::vector<int>& newShape) const
    {
        if (newShape.size() < shape.size())
        {
            printf("New shape must have at least as many dimensions\n");
            return nullptr;
        }
        // 检查是否可扩展
        int size_diff = (int)newShape.size() - (int)shape.size();
        for (auto i = 0; i < (int)shape.size(); ++i)
        {
            if (shape[i] != newShape[i + size_diff] && shape[i] != 1)
            {
                printf("Cannot expand dimension %d from %d to %d\n", i, shape[i], newShape[i + size_diff]);
                return nullptr;
            }
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        result->shape = newShape;
        result->strides.resize(newShape.size());
        for (int i = (int)newShape.size() - 1; i >= 0; --i)
        {
            int original_dim = i - size_diff;
            if (original_dim >= 0 && shape[original_dim] == 1)
            {
                result->strides[i] = 0;
            }
            else if (original_dim >= 0)
            {
                result->strides[i] = strides[original_dim];
            }
            else
            {
                result->strides[i] = 0;
            }
        }
        return result;
    }

    std::vector<int> GetContiguousStrides() const
    {
        return GetContiguousStrides(shape);
    }
    static std::vector<int> GetContiguousStrides(const std::vector<int>& shape)
    {
        std::vector<int> cont_strides(shape.size());
        if (shape.empty())
        {
            return cont_strides;
        }
        int stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i)
        {
            cont_strides[i] = stride;
            stride *= shape[i];
        }
        return cont_strides;
    }
    shared_ptr<T> GetData() const
    {
        return pdata;
	}
    std::vector<int> GetDimensionOrderByStride() const
    {
        std::vector<std::pair<int, int>> stride_pairs;
        for (auto i = 0; i < (int)strides.size(); ++i)

        {
            stride_pairs.push_back({ strides[i], i });
        }
        std::sort(stride_pairs.begin(), stride_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<int> order;
        for (const auto& p : stride_pairs)
        {
            order.push_back(p.second);
        }
        return order;
    }
    size_t GetOffset() const
    {
        return offset;
    }
    const std::vector<int> GetShape() const
    {
        return shape;
    }
    const std::vector<int> GetStrides() const
    {
        return strides;
	}
    size_t GetTotalElements() const
    {
        size_t total = 1;
        for (int dimSize : shape)
        {
            total *= dimSize;
        }
        return total;
    }
    static size_t GetTotalElements(const std::vector<int>& shape)
    {
        size_t total = 1;
        for (int dimSize : shape)
        {
            total *= dimSize;
        }
        return total;
    }
    virtual bool HasZero() const = 0;
    bool IsContiguous() const
    {
        if (strides.empty() || shape.empty())
            return true;
        size_t expect_stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i)
        {
            if (shape[i] == 1)
                continue;
            if (strides[i] != expect_stride)
                return false;
            expect_stride *= shape[i];
        }
        return true;
    }
    shared_ptr<Tensor> Narrow(const std::vector<int>& dims, const std::vector<int>& starts, const std::vector<int>& lengths) const
    {
        if (dims.size() != starts.size() || dims.size() != lengths.size())
        {
            printf("Narrow: dims, starts and lengths must have same size\n");
            return nullptr;
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        result->shape = this->shape;
        result->strides = this->strides;
        for (size_t i = 0; i < dims.size(); ++i)
        {
            int dim = dims[i];
            if (dim < 0) dim += (int)result->shape.size();
            int start = starts[i];
            if (start < 0) start += result->shape[dim];
            int length = lengths[i];
            if (dim < 0 || dim >= (int)result->shape.size())
            {
                printf("Narrow: Invalid dimension %d\n", dim);
                return nullptr;
            }
            if (start < 0 || start >= result->shape[dim])
            {
                printf("Narrow: Start index %d out of range\n", start);
                return nullptr;
            }
            if (length <= 0 || start + length > result->shape[dim])
            {
                printf("Narrow: Invalid length %d for dimension %d\n", length, dim);
                return nullptr;
            }
            result->offset += start * result->strides[dim];
            result->shape[dim] = length;
        }
        return result;
    }
    shared_ptr<Tensor> Narrow(int dim, int start, int length) const
    {
        if (dim < 0) dim += (int)shape.size();
        if (dim < 0 || dim >= (int)shape.size())
        {
            printf("Narrow: Invalid dimension %d\n", dim);
            return nullptr;
        }
        if (start < 0) start += shape[dim];
        if (start < 0 || start >= shape[dim])
        {
            printf("Narrow: Start index %d out of range [0, %d)\n", start, shape[dim]);
            return nullptr;
        }
        if (length <= 0)
        {
            printf("Narrow: Length must be positive\n");
            return nullptr;
        }
        if (start + length > shape[dim])
        {
            printf("Narrow: Slice [%d, %d) out of range for dimension size %d\n", start, start + length, shape[dim]);
            return nullptr;
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset + start * strides[dim];
        result->shape = this->shape;
        result->shape[dim] = length;
        result->strides = this->strides;
        return result;
    }
    bool Permute(const vector<int>& dims)
    {
        if (!CheckDims(dims))
            return false;
        auto oldShape = shape;
        auto oldStrides = strides;

        for (auto i = 0; i < (int)dims.size(); ++i)
        {
            auto dim = dims[i];
            if (dim < 0)
                dim = dim + (int)shape.size();
            shape[i] = oldShape[dim];
            strides[i] = oldStrides[dim];
        }
        return true;
    }
    void Print(bool pretty = true) const
    {
        if (shape.empty() || pdata == nullptr)
        {
            std::cout << "[]" << std::endl;
            return;
        }
		auto hostData = CopyDataToHost();
        std::vector<int> indices(shape.size(), 0);
        if (pretty)
            PrintPretty(0, indices, hostData, offset, shape, strides);
        else
            PrintCompact(0, indices, hostData, offset, shape, strides);
        std::cout << std::endl;
    }
    void PrintInfo() const
    {
        printf("Offset: %llu\n", offset);
        printf("IsContiguous: %s\n", IsContiguous() ? "True" : "False");
        printf("HasValidDenseLayout: %s\n", HasValidDenseLayout() ? "True" : "False");
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "Strides: [";
        for (size_t i = 0; i < strides.size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << strides[i];
        }
        std::cout << "]" << std::endl;
    }
    //Repeat 使用传入的repeats容器与原shape的各维度相乘得到新的shape。
    shared_ptr<Tensor> Repeat(const std::vector<int>& repeats) const
    {
        if (repeats.size() != shape.size())
        {
            return nullptr;
        }
        std::vector<int> newShape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i)
        {
            newShape[i] = shape[i] * repeats[i];
        }
        auto result = Create();
        int elements = 1;
        for (int s : newShape)
            elements *= s;

        result->pdata = Allocate(elements);
        result->offset = 0;
        result->shape = newShape;
        result->strides = result->GetContiguousStrides();

        std::vector<int> indices(shape.size(), 0);
        auto destData = result->pdata.get();
        RepeatRecursive(0, repeats, destData, indices);
        return result;
    }
    virtual void RepeatRecursive(int dim, const std::vector<int>& repeats, T*& destData, vector<int>& indices) const = 0;
    shared_ptr<Tensor> Reshape(const std::vector<int>& newShape) const
    {
        return Contiguous()->View(newShape);
    }
    shared_ptr<Tensor> Squeeze() const
    {
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        for (auto i = 0; i < (int)shape.size(); ++i)
        {
            if (shape[i] != 1)
            {
                result->shape.push_back(shape[i]);
                result->strides.push_back(strides[i]);
            }
        }
        // 如果所有维度都是1，则返回一个0-dim tensor（标量）
        if (result->shape.empty())
        {
            result->shape = { 1 };
            result->strides = { 1 };
        }
        return result;
    }
    shared_ptr<Tensor> Squeeze(const std::vector<int>& dims) const
    {
        std::vector<bool> to_squeeze(shape.size(), false);
        for (int dim : dims)
        {
            if (dim < 0) dim += (int)shape.size();
            if (dim < 0 || dim >= static_cast<int>(shape.size()))
            {
                printf("Invalid dimension for squeeze\n");
                return nullptr;
            }
            if (shape[dim] == 1)
            {
                to_squeeze[dim] = true;
            }
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (!to_squeeze[i])
            {
                result->shape.push_back(shape[i]);
                result->strides.push_back(strides[i]);
            }
        }
        return result;
    }
    shared_ptr<Tensor> Squeeze(int dim) const
    {
        if (dim < 0) dim += (int)shape.size();
        if (dim < 0 || dim >= static_cast<int>(shape.size()))
        {
            printf("Invalid dimension for squeeze\n");
            return nullptr;
        }
        auto result = Create();
        if (shape[dim] != 1)
        {
            *result = *this;
            return result;
        }
        result->pdata = this->pdata;
        result->offset = this->offset;
        for (int i = 0; i < (int)shape.size(); ++i)
        {
            if (i != dim)
            {
                result->shape.push_back(shape[i]);
                result->strides.push_back(strides[i]);
            }
        }
        return result;
    }
    bool Transpose(int dim1, int dim2)
    {
        if (dim1 < 0) dim1 += (int)shape.size();
        if (dim2 < 0) dim2 += (int)shape.size();
        if (dim1 < 0 || dim1 >= (int)shape.size() || dim2 < 0 || dim2 >= (int)shape.size())
        {
            return false;
        }
        if (dim1 == dim2)
        {
            return true;
        }
        std::swap(shape[dim1], shape[dim2]);
        std::swap(strides[dim1], strides[dim2]);
        return true;
    }
    shared_ptr<Tensor> Unsqueeze(int dim) const
    {
        if (dim < 0)
        {
            dim += (int)shape.size() + 1;  // +1 允许在最后插入
        }
        if (dim < 0 || dim > static_cast<int>(shape.size()))
        {
            printf("Invalid dimension for unsqueeze\n");
            return nullptr;
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        result->shape = this->shape;
        result->strides = this->strides;
        // 在指定位置插入维度1
        result->shape.insert(result->shape.begin() + dim, 1);
        // 计算新维度的步长
        if (dim == static_cast<int>(shape.size()))
        {
            // 在最后插入，步长为最后一个元素的步长
            if (shape.empty())
            {
                result->strides.insert(result->strides.begin() + dim, 1);
            }
            else
            {
                result->strides.insert(result->strides.begin() + dim, strides.empty() ? 1 : strides.back());
            }
        }
        else
        {
            // 在中间或开头插入，步长为当前维度的步长
            int new_stride;
            if (dim == 0)
            {
                // 在最前面插入，步长为第一个维度的步长乘以第一个维度的大小
                new_stride = shape.empty() ? 1 : strides[0] * shape[0];
            }
            else
            {
                // 在中间插入，步长为前一个维度的步长
                new_stride = strides[dim - 1];
            }
            result->strides.insert(result->strides.begin() + dim, new_stride);
        }
        return result;
    }
    shared_ptr<Tensor> View(const std::vector<int>& newShape) const
    {
        vector<int> cleanedShape = newShape;
        if (!ValidShape(cleanedShape))
        {
            return nullptr;
        }
        return ViewImpl(newShape);
    }

protected:
    uint64_t CalculateOffset(const std::vector<int>& indices) const
    {
        if (indices.size() != shape.size())
        {
            throw std::invalid_argument("Number of indices must match tensor dimensions");
        }
        uint64_t currOffset = this->offset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] < 0 || indices[i] >= shape[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            currOffset += indices[i] * strides[i];
        }
        return currOffset;
    }
    static uint64_t CalculateOffset(const std::vector<int>& indices, uint64_t offset, const std::vector<int>& shape, const std::vector<int>& strides)
    {
        if (indices.size() != shape.size())
        {
            throw std::invalid_argument("Number of indices must match tensor dimensions");
        }
        uint64_t currOffset = offset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] < 0 || indices[i] >= shape[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            currOffset += indices[i] * strides[i];
        }
        return currOffset;
	}
    uint64_t CalculateOffsetUnChecked(const std::vector<int>& indices) const
    {
        uint64_t curr_offset = this->offset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            curr_offset += indices[i] * strides[i];
        }
        return curr_offset;
    }
    bool CheckDevice(const Tensor& other) const
    {
        if (device != other.device)
        {
            throw std::invalid_argument("Tensor Device Do not match!");
        }
        return true;
    }
    bool CheckShape(const Tensor& other) const
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensor shapes Do not match!");
        }
        return true;
    }
    bool CheckIndex(const std::vector<int>& indices) const
    {
        if (indices.size() != shape.size())
            return false;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] < 0 || indices[i] >= shape[i])
                return false;
        }
        return true;
    }
    shared_ptr<Tensor> CloneContiguous() const
    {
        auto result = Create(shape);
        CopyDataContiguous(result->pdata.get());
        return result;
    }
    shared_ptr<Tensor> CloneDenseView() const
    {
        auto result = Create(shape);
        CopyDataContiguous(result->pdata.get());
        result->offset = 0;
        result->shape = shape;
        result->strides = strides;
        return result;
    }
    virtual void CopyDataContiguous(T* dest) const = 0;
    virtual void CopyDataNoncontiguous(T* dest) const = 0;
    bool HasValidDenseLayout() const
    {
        if (shape.empty() || !pdata)
        {
            return true;
        }
        for (int stride : strides)
        {
            if (stride <= 0)
            {
                return false;
            }
        }
        std::vector<int> dims(shape.size());
        std::iota(dims.begin(), dims.end(), 0);
        std::sort(dims.begin(), dims.end(), [&](int i, int j) {
            return strides[i] < strides[j];
            });
        size_t expectStride = 1;
        for (int dim : dims)
        {
            if (strides[dim] != expectStride)
            {
                return false;
            }
            expectStride = shape[dim] * strides[dim];
        }
        return true;
    }
    bool IsBroadcastable(const std::vector<int>& shape1, const std::vector<int>& shape2)
    {
        int ndim1 = shape1.size();
        int ndim2 = shape2.size();
        int minDim = std::min(ndim1, ndim2);

        for (int i = 1; i <= minDim; ++i)
        {
            int d1 = shape1[ndim1 - i];
            int d2 = shape2[ndim2 - i];
            if (d1 != d2 && d1 != 1 && d2 != 1)
                return false;
        }
        return true;
    }
    std::vector<int> GetBroadcastShape(const std::vector<int>& shape1, const std::vector<int>& shape2)
    {
        std::vector<int> out;
        int ndim1 = shape1.size();
        int ndim2 = shape2.size();
        int maxDim = std::max(ndim1, ndim2);

        for (int i = 1; i <= maxDim; ++i)
        {
            int d1 = (i <= ndim1) ? shape1[ndim1 - i] : 1;
            int d2 = (i <= ndim2) ? shape2[ndim2 - i] : 1;
            out.push_back(std::max(d1, d2));
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
    std::vector<size_t> BroadcastCoordinate(const std::vector<size_t>& outIndex, const std::vector<int>& srcShape)
    {
        std::vector<size_t> srcIndex;
        int offset = outIndex.size() - srcShape.size();
        for (int i = 0; i < srcShape.size(); ++i)
        {
            size_t dim = srcShape[i];
            size_t idx = outIndex[offset + i];
            srcIndex.push_back(dim == 1 ? 0 : idx);
        }
        return srcIndex;
    }
    std::vector<int> LinearToIndices(size_t linearIndex) const
    {
        std::vector<int> indices(shape.size());
        size_t remaining = linearIndex;
        for (int i = (int)shape.size() - 1; i >= 0; --i)
        {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }
        return indices;
    }
    static void PrintCompact(int dim, std::vector<int>& indices, const std::shared_ptr<T>& pdata, const size_t& offset, const std::vector<int>& shape, const std::vector<int>& strides)
    {
        std::cout << "[";

        if (dim == shape.size() - 1)
        {
            for (int i = 0; i < shape[dim]; ++i)
            {
                indices[dim] = i;
                size_t currOffset = CalculateOffset(indices, offset, shape, strides);
                std::cout << *(pdata.get() + currOffset);
                if (i < shape[dim] - 1)
                    std::cout << ", ";
            }
        }
        else
        {
            for (int i = 0; i < shape[dim]; ++i)
            {
                indices[dim] = i;
                PrintCompact(dim + 1, indices, pdata, offset, shape, strides);
                if (i < shape[dim] - 1)
                    std::cout << ", ";
            }
        }
        std::cout << "]";
    }
    static void PrintPretty(int dim, std::vector<int>& indices, const std::shared_ptr<T>& pdata, const size_t& offset, const std::vector<int>& shape, const std::vector<int>& strides)
    {
        std::cout << "[";
        if (dim == shape.size() - 1)
        {
            for (int i = 0; i < shape[dim]; ++i)
            {
                indices[dim] = i;
                if (i > 0)
                {
                    std::cout << ", ";
                    if (shape[dim] > 20 && i % 20 == 0)
                    {
                        std::cout << "\n";
                        for (int j = 0; j < dim + 1; ++j)
                            std::cout << " ";
                    }
                }
                size_t currOffset = CalculateOffset(indices, offset, shape, strides);
                std::cout << *(pdata.get() + currOffset);
            }
        }
        else
        {
            for (int i = 0; i < shape[dim]; ++i)
            {
                indices[dim] = i;
                if (i > 0)
                {
                    std::cout << "\n";
                    for (int j = 0; j < dim + 1; ++j)
                        std::cout << " ";
                }
                PrintPretty(dim + 1, indices, pdata, offset, shape, strides);
                if (i < shape[dim] - 1)
                {
                    std::cout << ",";
                }
            }
        }
        std::cout << "]";
    }
    void UpdateIndices(std::vector<int>& indices) const
    {
        int dim = (int)shape.size() - 1;
        while (dim >= 0)
        {
            indices[dim]++;
            if (indices[dim] < shape[dim])
            {
                break;
            }
            indices[dim] = 0;
            dim--;
        }
    }
    bool ValidShape(vector<int>& newShape) const
    {
        size_t newElements = 1;
        int minusIndex = -1;
        for (auto i = 0; i < (int)newShape.size(); ++i)
        {
            if (newShape[i] < -1)
            {
                printf("Dimensions must be positive\n");
                return false;
            }
            else if (newShape[i] == -1)
            {
                if (minusIndex != -1)
                {
                    printf("Only one dimension can be -1\n");
                    return false;
                }
                minusIndex = i;
            }
            else
            {
                newElements *= newShape[i];
            }
        }
        size_t elements = GetTotalElements();
        if (minusIndex != -1)
        {
            if (elements % newElements != 0)
            {
                printf("Shape is not divisible\n");
                return false;
            }
            newShape[minusIndex] = (int)(elements / newElements);
            return true;
        }
        else if (newElements != elements)
        {
            printf("New shape must have same number of elements\n");
            return false;
        }
        return true;
    }
    shared_ptr<Tensor> ViewImpl(const std::vector<int>& newShape) const
    {
        if (!IsContiguous())
        {
            return nullptr;
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        result->shape = newShape;
        result->strides.resize(newShape.size());
        int stride = 1;
        for (int i = (int)newShape.size() - 1; i >= 0; --i)
        {
            result->strides[i] = stride;
            stride *= newShape[i];
        }
        return result;
    }

protected:
    std::shared_ptr<T> pdata;
    uint64_t offset;
    vector<int> shape;
    vector<int> strides;
    DeviceType device;
};