#include "Tensor.h"


using namespace std;

template<typename T>
Tensor<T>::Tensor(DeviceType device) noexcept :pdata(nullptr), offset(0), device(device) {}
template<typename T>
Tensor<T>::Tensor(const vector<int>& shape, DeviceType device) noexcept :shape(shape), offset(0), device(device)
{
    this->strides = GetContiguousStrides(shape);
}
template<typename T>
Tensor<T>::Tensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape, DeviceType device) noexcept
    :pdata(data, deleter), offset(offset), shape(shape), device(device)
{
    strides = GetContiguousStrides(shape);
}
template<typename T>
Tensor<T>::Tensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape, const vector<int>& strides, DeviceType device) noexcept
	:pdata(data, deleter), offset(offset), shape(shape), strides(strides), device(device)
{
}
template<typename T>
Tensor<T>::Tensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, DeviceType device) noexcept
    :pdata(data), offset(offset), shape(shape), device(device)
{
    strides = GetContiguousStrides(shape);
}
template<typename T>
Tensor<T>::Tensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, const vector<int>& strides, DeviceType device) noexcept
    :pdata(data), offset(offset), shape(shape), strides(strides), device(device)
{
}
template<typename T>
Tensor<T>::Tensor(const Tensor& other) noexcept
{
    pdata = other.pdata;
    offset = other.offset;
    shape = other.shape;
    strides = other.strides;
    device = other.device;
}
template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
{
    pdata = std::move(other.pdata);
    offset = other.offset;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    device = other.device;
}
template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) noexcept
{
    pdata = other.pdata;
    offset = other.offset;
    shape = other.shape;
    strides = other.strides;
    device = other.device;
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        pdata = std::move(other.pdata);
        offset = other.offset;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        device = other.device;
    }
    return *this;
}
template<typename T>
template<typename... Indices>
T& Tensor<T>::operator()(Indices... indices)
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

template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator+(const Tensor& other) const
{
    return Add(other);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator-(const Tensor& other) const
{
    return Sub(other);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator*(const Tensor& other) const
{
    return Mul(other);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator/(const Tensor& other) const
{
    return Div(other);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator+(T value) const
{
    return Add(value);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator-(T value) const
{
    return Sub(value);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator*(T value) const
{
    return Mul(value);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::operator/(T value) const
{
    return Div(value);
}
template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor& other)
{
    Add_(other);
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor& other)
{
    Sub_(other);
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor& other)
{
    Mul_(other);
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor& other)
{
    Div_(other);
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator+=(T value)
{
    Add_(value);
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator-=(T value)
{
    Sub_(value);
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator*=(T value)
{
    Mul_(value);
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator/=(T value)
{
    Div_(value);
    return *this;
}


template<typename T>
uint64_t Tensor<T>::CalculateOffset(const std::vector<int>& indices) const
{
    if (indices.size() != shape.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    uint64_t currOffset = this->offset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] + shape[i] < 0 || indices[i] >= shape[i])
        {
            throw std::out_of_range("Index out of bounds");
        }
        if (indices[i] < 0)
        {
            currOffset += (indices[i] + shape[i]) * strides[i];
        }
        else
        {
            currOffset += indices[i] * strides[i];
        }
    }
    return currOffset;
}
template<typename T>
uint64_t Tensor<T>::CalculateOffset(const std::vector<int>& indices, uint64_t offset, const std::vector<int>& shape, const std::vector<int>& strides)
{
    if (indices.size() != shape.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    uint64_t currOffset = offset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] + shape[i] < 0 || indices[i] >= shape[i])
        {
            throw std::out_of_range("Index out of bounds");
        }
        if (indices[i] < 0)
        {
            currOffset += (indices[i] + shape[i]) * strides[i];
        }
        else
        {
            currOffset += indices[i] * strides[i];
        }
    }
    return currOffset;
}
template<typename T>
uint64_t Tensor<T>::CalculateOffsetUnChecked(const std::vector<int>& indices) const
{
    uint64_t curr_offset = this->offset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        curr_offset += indices[i] * strides[i];
    }
    return curr_offset;
}
template<typename T>
bool Tensor<T>::CheckDims(const vector<int>& dims)
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Clone() const
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Expand(const std::vector<int>& newShape) const
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
template<typename T>
std::vector<int> Tensor<T>::GetContiguousStrides() const
{
    return GetContiguousStrides(shape);
}
template<typename T>
std::vector<int> Tensor<T>::GetContiguousStrides(const std::vector<int>& shape)
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
template<typename T>
shared_ptr<T> Tensor<T>::GetData() const
{
    return pdata;
}
template<typename T>
DeviceType Tensor<T>::GetDevice() const
{
    return device;
}
template<typename T>
int Tensor<T>::GetDim() const
{
    return (int)shape.size();
}
template<typename T>
size_t Tensor<T>::GetOffset() const
{
    return offset;
}
template<typename T>
const std::vector<int> Tensor<T>::GetShape() const
{
    return shape;
}
template<typename T>
const std::vector<int> Tensor<T>::GetStrides() const
{
    return strides;
}
template<typename T>
size_t Tensor<T>::GetTotalElements() const
{
    size_t total = 1;
    for (int dimSize : shape)
    {
        total *= dimSize;
    }
    return total;
}
template<typename T>
size_t Tensor<T>::GetTotalElements(const std::vector<int>& shape)
{
    size_t total = 1;
    for (int dimSize : shape)
    {
        total *= dimSize;
    }
    return total;
}
template<typename T>
bool Tensor<T>::HasValidDenseLayout() const
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
template<typename T>
bool Tensor<T>::IsContiguous() const
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Narrow(const std::vector<int>& dims, const std::vector<int>& starts, const std::vector<int>& lengths) const
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Narrow(int dim, int start, int length) const
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
template<typename T>
bool Tensor<T>::Permute(const vector<int>& dims)
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
template<typename T>
void Tensor<T>::Print(bool pretty) const
{
    if (shape.empty() || pdata == nullptr)
    {
        std::cout << "[]" << std::endl;
        return;
    }
    if (this->device != DeviceType::CPU)
    {
        this->ToDevice(DeviceType::CPU)->Print(pretty);
        return;
	}
    else
    {
        std::vector<int> indices(shape.size(), 0);
        if (pretty)
        {
            PrintPrettyRecursive(0, pdata.get() + offset, shape, strides);
        }
        else
        {
            PrintCompactRecursive(0, pdata.get() + offset, shape, strides);
        }
        std::cout << std::endl;
    }
}
template<typename T>
void Tensor<T>::PrintInfo() const
{
    printf("Offset: %llu\n", (unsigned long long)offset);
    printf("IsContiguous: %s\n", IsContiguous() ? "True" : "False");
    printf("HasValidDenseLayout: %s\n", HasValidDenseLayout() ? "True" : "False");
	printf("Device: %s\n", device == DeviceType::CPU ? "CPU" : "CUDA");
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Repeat(int dim, int repeat) const
{
    if (dim < 0) dim += (int)shape.size();
    if (dim < 0 || dim >= (int)shape.size())
    {
        printf("Invalid dimension for repeat\n");
        return nullptr;
    }
    if (repeat <= 0)
    {
        printf("Repeat count must be positive\n");
        return nullptr;
	}
    std::vector<int> repeats(shape.size(), 1);
    repeats[dim] = repeat;
    return Repeat(repeats);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Repeat(const std::vector<int>& repeats) const
{
    if (repeats.size() != shape.size())
    {
        return nullptr;
    }
    for (auto r : repeats)
    {
        if (r <= 0)
        {
            printf("Repeats count must be positive\n");
            return nullptr;
        }
	}
    std::vector<int> newShape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        newShape[i] = shape[i] * repeats[i];
    }
    auto result = Create(newShape);
	CopyDataRepeat(result.get());
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Reshape(const std::vector<int>& newShape) const
{
    return Contiguous()->View(newShape);
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Squeeze() const
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Squeeze(int dim) const
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Squeeze(const std::vector<int>& dims) const
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
template<typename T>
bool Tensor<T>::Transpose(int dim1, int dim2)
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::Unsqueeze(int dim) const
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
template<typename T>
void Tensor<T>::UpdateIndices(std::vector<int>& indices) const
{
    for (int dim = (int)shape.size() - 1; dim >= 0; --dim)
    {
        if (++indices[dim] < shape[dim])
        {
            break;
        }
        else
        {
            indices[dim] = 0;
        }
    }
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::View(const std::vector<int>& newShape) const
{
    vector<int> cleanedShape = newShape;
    if (!ValidShape(cleanedShape))
    {
        return nullptr;
    }
    return ViewImpl(newShape);
}


template<typename T>
bool Tensor<T>::CheckDevice(const Tensor& other) const
{
    if (device != other.device)
    {
        throw std::invalid_argument("Tensor Device Do not match!");
    }
    return true;
}
template<typename T>
bool Tensor<T>::CheckShape(const Tensor& other) const
{
    if (shape != other.shape)
    {
        throw std::invalid_argument("Tensor shapes Do not match!");
    }
    return true;
}
template<typename T>
bool Tensor<T>::CheckIndex(const std::vector<int>& indices) const
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::CloneContiguous() const
{
    auto result = Create(shape);
    CopyDataContiguous(result->pdata.get());
    return result;
}
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::CloneDenseView() const
{
    auto result = Create(shape);
    CopyDataContiguous(result->pdata.get());
    result->offset = 0;
    result->shape = shape;
    result->strides = strides;
    return result;
}
template<typename T>
std::vector<int> Tensor<T>::LinearToIndices(size_t linearIndex) const
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
template<typename T>
void Tensor<T>::PrintCompactRecursive(int dim, T* data, const std::vector<int>& shape, const std::vector<int>& strides)
{
    std::cout << "[";
    if (dim == shape.size() - 1)
    {
        for (int i = 0; i < shape[dim]; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << *(data + i * strides[dim]);
        }
    }
    else
    {
        for (int i = 0; i < shape[dim]; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            PrintCompactRecursive(dim + 1, data + i * strides[dim], shape, strides);
        }
    }
    std::cout << "]";
}
template<typename T>
void Tensor<T>::PrintPrettyRecursive(int dim, T* data, const std::vector<int>& shape, const std::vector<int>& strides)
{
    std::cout << "[";
    if (dim == shape.size() - 1)
    {
        for (int i = 0; i < shape[dim]; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << *(data + i * strides[dim]);
        }
    }
    else
    {
        for (int i = 0; i < shape[dim]; ++i)
        {
            if (i > 0)
                std::cout << ",\n" << std::string(dim + 1, ' ');
            PrintPrettyRecursive(dim + 1, data + i * strides[dim], shape, strides);
        }
    }

    std::cout << "]";
}
template<typename T>
bool Tensor<T>::ValidShape(vector<int>& newShape) const
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
template<typename T>
shared_ptr<Tensor<T>> Tensor<T>::ViewImpl(const std::vector<int>& newShape) const
{
    if (!IsContiguous())
    {
        return nullptr;
    }
    auto result = Create();
    result->pdata = this->pdata;
    result->offset = this->offset;
    result->shape = newShape;
    result->strides = result->GetContiguousStrides();
    return result;
}


template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int8_t>;
template class Tensor<uint8_t>;
template class Tensor<int16_t>;
template class Tensor<uint16_t>;
template class Tensor<int32_t>;
template class Tensor<uint32_t>;
template class Tensor<int64_t>;
template class Tensor<uint64_t>;
