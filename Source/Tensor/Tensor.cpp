#include "Tensor.h"


Tensor::Tensor()
    :pdata(nullptr), data_len(0), offset(0)
{
}
Tensor::Tensor(byte* data, unsigned int data_len, unsigned int offset, const vector<int>& shape, const vector<int>& strides)
    :pdata(data), data_len(data_len), offset(offset), shape(shape), strides(strides)
{

}
bool Tensor::check_dims(const vector<int>& dims)
{
    if (dims.size() != this->shape.size())
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
bool Tensor::permute(const vector<int>& dims)
{
    if (!check_dims(dims))
        return false;
    auto oldShape = shape;
    auto oldStrides = strides;

    for (auto i = 0; i < dims.size(); ++i)
    {
        auto dim = dims[i];
        if (dim < 0)
            dim = dim + (int)shape.size();
        shape[i] = oldShape[dim];
        strides[i] = oldStrides[dim];
    }
    return true;
}
bool Tensor::transpose(int dim1, int dim2)
{
    if (dim1 < 0) dim1 += (int)shape.size();
    if (dim2 < 0) dim2 += (int)shape.size();

    if (dim1 < 0 || dim1 >= (int)shape.size() || dim2 < 0 || dim2 >= (int)shape.size())
    {
        throw std::out_of_range("Invalid dimension for transpose");
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
bool Tensor::is_contiguous() const
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
std::vector<int> Tensor::get_contiguous_strides() const
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
void Tensor::copy_recursive(int dim, std::vector<int>& indices, int*& new_ptr) const
{
    if (dim == shape.size())
    {
        int offset = this->offset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            offset += indices[i] * strides[i];
        }
        int value = *((int*)pdata.get() + offset);
        *new_ptr++ = value;
        return;
    }
    for (int i = 0; i < shape[dim]; ++i) {
        indices[dim] = i;
        copy_recursive(dim + 1, indices, new_ptr);
    }
}
void Tensor::contiguous_inplace()
{
    if (is_contiguous()) return;

    int total_elements = get_total_elements();
    int new_data_len = total_elements * sizeof(int);

    std::shared_ptr<byte[]> new_pdata(new byte[new_data_len]);
    std::vector<int> cont_strides = get_contiguous_strides();
    std::vector<int> indices(shape.size(), 0);
    int* new_ptr = (int*)new_pdata.get();
    copy_recursive(0, indices, new_ptr);

    pdata = new_pdata;
    data_len = new_data_len;
    offset = 0;
    strides = std::move(cont_strides);
}
Tensor Tensor::contiguous() const
{
    Tensor result;
    if (is_contiguous())
    {
        result.pdata = this->pdata;
        result.data_len = this->data_len;
        result.offset = this->offset;
        result.shape = this->shape;
        result.strides = this->strides;
        return result;
    }
    int total_elements = get_total_elements();
    result.data_len = total_elements * sizeof(int);
    result.pdata = std::shared_ptr<byte[]>(new byte[result.data_len]);
    result.offset = 0;
    result.shape = this->shape;
    result.strides = get_contiguous_strides();
    std::vector<int> indices(shape.size(), 0);
    int* dest_ptr = (int*)result.pdata.get();
    copy_recursive(0, indices, dest_ptr);
    return result;
}
std::vector<int> Tensor::get_dimension_order_by_stride() const
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

int& Tensor::at(const std::vector<int>& indices)
{
    int offset = calculate_offset(indices);
    return *((int*)(pdata.get()) + offset);
}
const int& Tensor::at(const std::vector<int>& indices) const
{
    int offset = calculate_offset(indices);
    return *((int*)(pdata.get()) + offset);
}
int Tensor::calculate_offset(const std::vector<int>& indices) const
{
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }

    int offset = this->offset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] < 0 || indices[i] >= shape[i])
        {
            throw std::out_of_range("Index out of bounds");
        }
        offset += indices[i] * strides[i];
    }
    return offset;
}
void Tensor::update_indices(std::vector<int>& indices) const
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
int Tensor::get_total_elements() const
{
    int total = 1;
    for (int dim_size : shape)
    {
        total *= dim_size;
    }
    return total;
}
void Tensor::for_each_linear(const std::function<void(int&, const std::vector<int>&)>& func)
{
    int total_elements = get_total_elements();
    std::vector<int> indices(shape.size(), 0);
    for (int i = 0; i < total_elements; ++i)
    {
        // 使用strides计算正确的内存位置
        int& value = at(indices);
        func(value, indices);
        update_indices(indices);
    }
}
void Tensor::for_each_batch(const std::function<void(int*, int, int)>& func)
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
            int* batch_start = (int*)(pdata.get()) + offset;
            func(batch_start, inner_size, inner_stride);
            return;
        }
        for (int i = 0; i < shape[depth]; ++i)
        {
            outer_indices[depth] = i;
            traverse_outer(depth + 1);
        }};
    traverse_outer(0);
}
void Tensor::for_each_batch_with_indices(const std::function<void(int*, int, int, const std::vector<int>&)>& func)
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
            int* batch_start = (int*)(pdata.get()) + offset;
            func(batch_start, inner_size, inner_stride, outer_indices);
            return;
        }
        for (int i = 0; i < shape[depth]; ++i) {
            outer_indices[depth] = i;
            traverse_outer(depth + 1);
        }};
    traverse_outer(0);
}
void Tensor::print_info() const
{
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
    printf("is_contiguous:%d\n", is_contiguous());
}

