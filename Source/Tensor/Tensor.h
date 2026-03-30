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

template<typename T>
class Tensor
{
public:
    Tensor() noexcept :pdata(nullptr), offset(0) {}
    Tensor(T* data, uint64_t offset, const vector<int>& shape, const vector<int>& strides) noexcept
        :pdata(data, [](T* p) { delete[] p; }), offset(offset), shape(shape), strides(strides)
    {}
    Tensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, const vector<int>& strides) noexcept
        :pdata(data), offset(offset), shape(shape), strides(strides)
    {}
    
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
    Tensor operator-() const
    {
        auto result = Create(shape);
        auto it = begin();
        auto resultIt = result->begin();
        while (it != end())
        {
            *resultIt = -*it;
            ++it;
            ++resultIt;
        }
        return *result;
    }
    Tensor operator+(const Tensor& other) const
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
        auto result = Create(shape);
        auto it1 = begin();
        auto it2 = other.begin();
        auto resultIt = result->begin();
        while (it1 != end())
        {
            *resultIt = *it1 + *it2;
            ++it1;
            ++it2;
            ++resultIt;
        }
        return *result;
    }
    Tensor operator-(const Tensor& other) const
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }
        auto result = Create(shape);
        auto it1 = begin();
        auto it2 = other.begin();
        auto resultIt = result->begin();
        while (it1 != end())
        {
            *resultIt = *it1 - *it2;
            ++it1;
            ++it2;
            ++resultIt;
        }
        return *result;
    }
    Tensor operator*(T value) const
    {
        auto result = Create(shape);
        auto it = begin();
        auto resultIt = result->begin();
        while (it != end())
        {
            *resultIt = *it * value;
            ++it;
            ++resultIt;
        }
        return *result;
    }
    Tensor operator*(const Tensor& other) const
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
        }
        auto result = Create(shape);
        auto it1 = begin();
        auto it2 = other.begin();
        auto resultIt = result->begin();
        while (it1 != end())
        {
            *resultIt = *it1 * *it2;
            ++it1;
            ++it2;
            ++resultIt;
        }
        return *result;
    }
    Tensor operator/(T value) const
    {
        if (value == T(0))
        {
            throw std::invalid_argument("Division by zero");
        }
        auto result = Create(shape);
        auto it = begin();
        auto resultIt = result->begin();
        while (it != end())
        {
            *resultIt = *it / value;
            ++it;
            ++resultIt;
        }
        return *result;
    }
    Tensor& operator+=(const Tensor& other)
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
        auto it1 = begin();
        auto it2 = other.begin();
        while (it1 != end())
        {
            *it1 = *it1 + *it2;
            ++it1;
            ++it2;
        }
        return *this;
    }
    Tensor& operator*=(T value)
    {
        auto it = begin();
        while (it != end())
        {
            *it = *it * value;
            ++it;
        }
        return *this;
    }
    bool operator==(const Tensor& other) const
    {
        if (shape != other.shape)
        {
            return false;
        }
        auto it1 = begin();
        auto it2 = other.begin();
        while (it1 != end())
        {
            if (*it1 != *it2)
            {
                return false;
            }
            ++it1;
            ++it2;
        }
        return true;
    }
    bool operator!=(const Tensor& other) const
    {
        return !(*this == other);
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
        iterator(Tensor* tensor, size_t linear_idx)
            : tensor(tensor), linearIndex(linear_idx)
        {}
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
        Tensor* tensor;
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
        const_iterator(const Tensor* tensor, size_t linear_idx)
            : tensor(tensor), linearIndex(linear_idx)
        {}
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
        const Tensor* tensor;
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
        return iterator(this, GetTotalElements());
    }
    const_iterator end() const
    {
        return const_iterator(this, GetTotalElements());
    }
    const_iterator cend() const
    {
        return const_iterator(this, GetTotalElements());
    }


    std::shared_ptr<T> Allocate(size_t newSize) const
    {
        return shared_ptr<T>(new T[newSize], [](T* p) {delete[] p; });
    }
    const T& At(const std::vector<int>& indices, bool checkBounds = true) const
    {
        size_t curr_offset = checkBounds ? CalculateOffset(indices) : CalculateOffsetUnChecked(indices);
        return *(pdata.get() + curr_offset);
    }
    T& At(const std::vector<int>& indices, bool checkBounds = true)
    {
        size_t curr_offset = checkBounds ? CalculateOffset(indices) : CalculateOffsetUnChecked(indices);
        return *(pdata.get() + curr_offset);
    }
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
    shared_ptr<Tensor> Contiguous() const
    {
        auto result = Create();
        if (IsContiguous())
        {
            *result = *this;
            return shared_ptr<Tensor>(result);
        }
        size_t newSize = GetTotalElements();
        result->pdata = Allocate(newSize);
        result->offset = 0;
        result->shape = this->shape;
        result->strides = GetContiguousStrides();
        std::vector<int> indices(shape.size(), 0);
        auto dest_ptr = result->pdata.get();
        CopyRecursive(0, indices, dest_ptr);
        return result;
    }
    void ContiguousInplace()
    {
        if (IsContiguous()) return;

        size_t newSize = GetTotalElements();
        auto new_pdata = Allocate(newSize);
        std::vector<int> cont_strides = GetContiguousStrides();
        std::vector<int> indices(shape.size(), 0);
        auto new_ptr = new_pdata.get();
        CopyRecursive(0, indices, new_ptr);

        pdata = new_pdata;
        offset = 0;
        strides = std::move(cont_strides);
    }
    static shared_ptr<Tensor> Create()
    {
        return make_shared<Tensor>();
    }
    static shared_ptr<Tensor> Create(const std::vector<int>& shape)
    {
		auto tensor = make_shared<Tensor>();
		tensor->shape = shape;
		tensor->strides = GetContiguousStrides(shape);
        size_t total = tensor->GetTotalElements();
        tensor->pdata = std::shared_ptr<T>(new T[total], std::default_delete<T[]>());
        tensor->offset = 0;
        return tensor;
    }
    T* Data()
    {
        return pdata.get() + offset;
	}
    int Dim() const
    {
        return (int)shape.size();
	}
    shared_ptr<Tensor> Expand(const std::vector<int>& new_shape) const
    {
        if (new_shape.size() < shape.size())
        {
            printf("New shape must have at least as many dimensions\n");
            return nullptr;
        }
        // 检查是否可扩展
        int size_diff = (int)new_shape.size() - (int)shape.size();
        for (auto i = 0; i < (int)shape.size(); ++i)
        {
            if (shape[i] != new_shape[i + size_diff] && shape[i] != 1)
            {
                printf("Cannot expand dimension %d from %d to %d\n", i, shape[i], new_shape[i + size_diff]);
                return nullptr;
            }
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        result->shape = new_shape;
        result->strides.resize(new_shape.size());
        // 计算新步长
        for (int i = (int)new_shape.size() - 1; i >= 0; --i)
        {
            int original_dim = i - size_diff;
            if (original_dim >= 0 && shape[original_dim] == 1)
            {
                // 原始维度为1，可以广播，步长为0
                result->strides[i] = 0;
            }
            else if (original_dim >= 0)
            {
                // 普通维度，使用原始步长
                result->strides[i] = strides[original_dim];
            }
            else
            {
                // 新增的维度，步长设为0（可以广播）
                result->strides[i] = 0;
            }
        }
        return result;
    }
    void ForEachBatch(const std::function<void(T*, int, int)>& func) const
    {
        if (shape.empty()) return;
        int inner_dim = (int)shape.size() - 1;
        int inner_size = shape[inner_dim];
        int inner_stride = strides[inner_dim];

        std::vector<int> outer_indices(shape.size() - 1, 0);
        std::function<void(int)> traverse_outer = [&](int depth) {
            if (depth == static_cast<int>(shape.size() - 1))
            {
                int offset = this->offset;
                for (size_t i = 0; i < outer_indices.size(); ++i)
                {
                    offset += outer_indices[i] * strides[i];
                }
                T* batch_start = pdata.get() + offset;
                func(batch_start, inner_size, inner_stride);
                return;
            }
            for (int i = 0; i < shape[depth]; ++i)
            {
                outer_indices[depth] = i;
                traverse_outer(depth + 1);
            }
            };
        traverse_outer(0);
    }
    void ForEachLinear(const std::function<void(T, const std::vector<int>&)>& func) const
    {
        size_t total_elements = GetTotalElements();
        std::vector<int> indices(shape.size(), 0);
        for (size_t i = 0; i < total_elements; ++i)
        {
            // 使用strides计算正确的内存位置
            auto value = At(indices, false);
            func(value, indices);
            UpdateIndices(indices);
        }
    }
    static shared_ptr<Tensor> Full(const std::vector<int>& shape, T value)
    {
        auto tensor = Create(shape);
        T* data_ptr = tensor->pdata.get();
        size_t total = tensor->GetTotalElements();
        if (value == T(0))
        {
            std::memset(data_ptr, 0, total * sizeof(T));
        }
        else
        {
            std::fill(data_ptr, data_ptr + total, value);
        }
        return tensor;
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
    static shared_ptr<Tensor> Ones(const std::vector<int>& shape)
    {
        auto tensor = Create(shape);
        T* data_ptr = tensor->pdata.get();
        size_t total = tensor->GetTotalElements();
        std::fill(data_ptr, data_ptr + total, T(1));
        return tensor;
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
        std::vector<int> indices(shape.size(), 0);
        if (pretty)
            PrintPretty(0, indices);
        else
            PrintCompact(0, indices);
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
        std::vector<int> new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i)
        {
            new_shape[i] = shape[i] * repeats[i];
        }
        auto result = Create();
        int total_elements = 1;
        for (int s : new_shape)
            total_elements *= s;

        result->pdata = Allocate(total_elements);
        result->offset = 0;
        result->shape = new_shape;
        result->strides = result->GetContiguousStrides();

        std::vector<int> indices(shape.size(), 0);
        std::vector<int> new_indices(new_shape.size(), 0);
        auto dest_ptr = result->pdata.get();
        std::function<void(int)> repeat_recursive = [&](int dim) {
            if (dim == shape.size() - 1)
            {
                size_t curr_offset = this->offset;
                for (size_t i = 0; i < indices.size(); ++i)
                {
                    curr_offset += indices[i] * strides[i];
                }
                auto dataStart = pdata.get() + curr_offset;
                for (auto j = 0; j < shape[dim]; ++j)
                {
                    *dest_ptr++ = *(dataStart + j * strides[dim]);
                }
                return;
            }
            for (int i = 0; i < repeats[dim]; ++i)
            {
                for (int j = 0; j < shape[dim]; ++j)
                {
                    indices[dim] = j;
                    repeat_recursive(dim + 1);
                }
            }
            };
        repeat_recursive(0);
        return result;
    }
    shared_ptr<Tensor> Reshape(const std::vector<int>& new_shape) const
    {
        return Contiguous()->View(new_shape);
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
    size_t Size() const
    {
		return GetTotalElements();
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
    shared_ptr<Tensor> View(const std::vector<int>& new_shape) const
    {
        size_t new_total = 1;
        int minus_one_idx = -1;
        for (auto i = 0; i < (int)new_shape.size(); ++i)
        {
            if (new_shape[i] <= 0)
            {
                if (new_shape[i] == -1)
                {
                    if (minus_one_idx != -1)
                    {
                        printf("Only one dimension can be -1\n");
                        return nullptr;
                    }
                    minus_one_idx = i;
                }
                else
                {
                    printf("Dimensions must be positive\n");
                    return nullptr;
                }
            }
            new_total *= new_shape[i];
        }
        size_t old_total = GetTotalElements();
        if (minus_one_idx != -1)
        {
            if (old_total % new_total != 0)
            {
                printf("Shape is not divisible\n");
                return nullptr;
            }
            std::vector<int> final_shape = new_shape;
            final_shape[minus_one_idx] = (int)(old_total / new_total);
            return ViewImpl(final_shape);
        }
        else
        {
            if (new_total != old_total)
            {
                printf("New shape must have same number of elements\n");
                return nullptr;
            }
            return ViewImpl(new_shape);
        }
    }
    static shared_ptr<Tensor> Zeros(const std::vector<int>& shape)
    {
        auto tensor = Create(shape);
        T* data_ptr = tensor->pdata.get();
        size_t total_bytes = tensor->GetTotalElements() * sizeof(T);
        std::memset(data_ptr, 0, total_bytes);
        return tensor;
    }
private:
    uint64_t CalculateOffset(const std::vector<int>& indices) const
    {
        if (indices.size() != shape.size())
        {
            throw std::invalid_argument("Number of indices must match tensor dimensions");
        }
        uint64_t curr_offset = this->offset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] < 0 || indices[i] >= shape[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            curr_offset += indices[i] * strides[i];
        }
        return curr_offset;
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
        auto result = Create();
		auto elementCount = GetTotalElements();
        result->pdata = std::shared_ptr<T>(new T[elementCount], std::default_delete<T[]>());
        std::memcpy(result->pdata.get(), pdata.get() + offset, elementCount * sizeof(T));
        result->offset = 0;
        result->shape = shape;
        result->strides = strides;
        return result;
    }
    shared_ptr<Tensor> CloneDenseView() const
    {
        auto result = Create();
        auto elementCount = GetTotalElements();
        result->pdata = std::shared_ptr<T>(new T[elementCount], std::default_delete<T[]>());
        std::memcpy(result->pdata.get(), pdata.get() + offset, elementCount * sizeof(T));
        result->offset = 0;
        result->shape = shape;
        result->strides = strides;
        return result;
    }
    void CopyData(shared_ptr<Tensor> other) const
    {
        // 遍历所有元素，按原步长读取，按新步长写入
        std::function<void(size_t, size_t, size_t)> copyRecursive =
            [&](size_t dim, size_t srcOffset, size_t destOffset) {
            if (dim == shape.size() - 1)
            {
                for (auto i = 0; i < shape[dim]; ++i)
                {
                    other->pdata.get()[destOffset] = pdata.get()[srcOffset];
                    ++destOffset;
                    srcOffset += strides[dim];
                }
                
                return;
            }
            for (int i = 0; i < shape[dim]; i++)
            {
                copyRecursive(dim + 1, srcOffset + i * strides[dim], destOffset + i * other->strides[dim]);
            }};

        copyRecursive(0, offset, 0);
    }
    void CopyRecursive(int dim, std::vector<int>& indices, T*& new_ptr, size_t currOffset = 0) const
    {
        if (dim == shape.size() - 1)
        {
            T* currData = pdata.get() + this->offset + currOffset;
            for (auto i = 0; i < shape[dim]; ++i)
            {
                *new_ptr++ = *currData;
                currData += strides[dim];
            }
            return;
        }
        for (int i = 0; i < shape[dim]; ++i)
        {
            indices[dim] = i;
			size_t newOffset = currOffset + i * strides[dim];
            CopyRecursive(dim + 1, indices, new_ptr, newOffset);
        }
    }
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
    void PrintCompact(int dim, std::vector<int>& indices) const
    {
        std::cout << "[";

        if (dim == shape.size() - 1)
        {
            for (int i = 0; i < shape[dim]; ++i)
            {
                indices[dim] = i;
                size_t currOffset = CalculateOffset(indices);
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
                PrintCompact(dim + 1, indices);
                if (i < shape[dim] - 1)
                    std::cout << ", ";
            }
        }
        std::cout << "]";
    }
    void PrintPretty(int dim, std::vector<int>& indices) const
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
                size_t currOffset = CalculateOffset(indices);
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
                PrintPretty(dim + 1, indices);
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
    shared_ptr<Tensor> ViewImpl(const std::vector<int>& new_shape) const
    {
        if (!IsContiguous())
        {
            return nullptr;
        }
        auto result = Create();
        result->pdata = this->pdata;
        result->offset = this->offset;
        result->shape = new_shape;
        result->strides.resize(new_shape.size());
        int stride = 1;
        for (int i = (int)new_shape.size() - 1; i >= 0; --i)
        {
            result->strides[i] = stride;
            stride *= new_shape[i];
        }
        return result;
    }

private:
    std::shared_ptr<T> pdata;
    uint64_t offset;
    vector<int> shape;
    vector<int> strides;
};