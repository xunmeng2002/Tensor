#include <string>
#include <iostream>
#include <vector>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <memory>

using namespace std;

template<typename T>
class Tensor
{
public:
    Tensor() :pdata(nullptr), offset(0) {}
    Tensor(T* data, unsigned int offset, const vector<int>& shape, const vector<int>& strides)
        :pdata(data, [](T* p) { delete[] p; }), offset(offset), shape(shape), strides(strides)
    {}
    Tensor(shared_ptr<T> data, unsigned int offset, const vector<int>& shape, const vector<int>& strides)
        :pdata(data), offset(offset), shape(shape), strides(strides)
    {
    }

    Tensor(const Tensor& other)
    {
        pdata = other.pdata;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;
    }
    Tensor& operator=(const Tensor& other)
    {
        pdata = other.pdata;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;
        return *this;
    }
    Tensor(Tensor&& other)
    {
        pdata = std::move(other.pdata);
        offset = other.offset;
        shape = std::move(other.shape);
		strides = std::move(other.strides);
    }
    Tensor& operator=(Tensor&& other)
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

    std::shared_ptr<T> Allocate(int newSize) const
    {
        return shared_ptr<T>(new T[newSize], [](T* p) {delete[] p; });
    }
    T At(const std::vector<int>& indices) const
    {
        int curr_offset = CalculateOffset(indices);
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
    shared_ptr<Tensor> Contiguous() const
    {
        auto result = Create();
        if (IsContiguous())
        {
            *result = *this;
            return shared_ptr<Tensor>(result);
        }
        int newSize = GetTotalElements();
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

        auto newSize = GetTotalElements();
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
        int total = 1;
        for (int s : shape)
            total *= s;
		auto tensor = make_shared<Tensor>();
        tensor->pdata = std::shared_ptr<T>(new T[total], std::default_delete<T[]>());
		tensor->offset = 0;
		tensor->shape = shape;
		tensor->strides = GetContiguousStrides(shape);
        return tensor;
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
    void ForEachBatch(const std::function<void(T*, int, int)>& func)
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
    void ForEachBatchWithIndices(const std::function<void(T*, int, int, const std::vector<int>&)>& func)
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
                func(batch_start, inner_size, inner_stride, outer_indices);
                return;
            }
            for (int i = 0; i < shape[depth]; ++i) {
                outer_indices[depth] = i;
                traverse_outer(depth + 1);
            }};
        traverse_outer(0);
    }
    void ForEachLinear(const std::function<void(T, const std::vector<int>&)>& func)
    {
        int total_elements = GetTotalElements();
        std::vector<int> indices(shape.size(), 0);
        for (int i = 0; i < total_elements; ++i)
        {
            // 使用strides计算正确的内存位置
            auto value = At(indices);
            func(value, indices);
            UpdateIndices(indices);
        }
    }
    static shared_ptr<Tensor> Full(const std::vector<int>& shape, T value)
    {
        auto tensor = Create(shape);
        T* data_ptr = tensor->pdata.get();
        int total = tensor->GetTotalElements();
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
    bool IsContiguous() const
    {
        if (strides.empty() || shape.empty())
            return true;
        if (offset != 0)
            return false;
        int expect_stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i)
        {
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
        int total = tensor->GetTotalElements();
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
        printf("Offset: %u\n", offset);
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
        printf("is_contiguous:%d\n", IsContiguous());
    }
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
                int curr_offset = this->offset;
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
        int new_total = 1;
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
        int old_total = GetTotalElements();
        if (minus_one_idx != -1)
        {
            // 计算-1维度的大小
            if (old_total % new_total != 0)
            {
                printf("Shape is not divisible\n");
                return nullptr;
            }
            std::vector<int> final_shape = new_shape;
            final_shape[minus_one_idx] = old_total / new_total;
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
        int total_bytes = tensor->GetTotalElements() * sizeof(T);
        std::memset(data_ptr, 0, total_bytes);
        return tensor;
    }
private:
    int CalculateOffset(const std::vector<int>& indices) const
    {
        if (indices.size() != shape.size())
        {
            throw std::invalid_argument("Number of indices must match tensor dimensions");
        }
        int curr_offset = this->offset;
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
    void CopyRecursive(int dim, std::vector<int>& indices, T*& new_ptr) const
    {
        if (dim == shape.size() - 1)
        {
            int curr_offset = this->offset;
            for (size_t i = 0; i < dim; ++i)
            {
                curr_offset += indices[i] * strides[i];
            }
            T* currData = pdata.get() + curr_offset;
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
            CopyRecursive(dim + 1, indices, new_ptr);
        }
    }
    int GetTotalElements() const
    {
        int total = 1;
        for (int dim_size : shape)
        {
            total *= dim_size;
        }
        return total;
    }
    void PrintCompact(int dim, std::vector<int>& indices) const
    {
        std::cout << "[";

        if (dim == shape.size() - 1)
        {
            for (int i = 0; i < shape[dim]; ++i)
            {
                indices[dim] = i;
                int offset = CalculateOffset(indices);
                std::cout << *(pdata.get() + offset);
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
                int offset = CalculateOffset(indices);
                std::cout << *(pdata.get() + offset);
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
    unsigned int offset;
    vector<int> shape;
    vector<int> strides;
};