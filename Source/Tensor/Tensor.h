#include <iostream>
#include <vector>
#include <cstddef>
#include <functional>
#include <algorithm>

using namespace std;

class Tensor
{
public:
    Tensor();
    Tensor(byte* data, unsigned int data_len, unsigned int offset, const vector<int>& shape, const vector<int>& strides);
    bool check_dims(const vector<int>& dims);
    bool permute(const vector<int>& dims);
    bool transpose(int dim1, int dim2);
    bool is_contiguous() const;
    std::vector<int> get_contiguous_strides() const;
    void copy_recursive(int dim, std::vector<int>& indices, int*& new_ptr) const;
    void contiguous_inplace();
    Tensor contiguous() const;
    std::vector<int> get_dimension_order_by_stride() const;

    int& at(const std::vector<int>& indices);
    const int& at(const std::vector<int>& indices) const;
    int calculate_offset(const std::vector<int>& indices) const;
    void update_indices(std::vector<int>& indices) const;
    int get_total_elements() const;
    void for_each_linear(const std::function<void(int&, const std::vector<int>&)>& func);
    void for_each_batch(const std::function<void(int*, int, int)>& func);
    void for_each_batch_with_indices(const std::function<void(int*, int, int, const std::vector<int>&)>& func);
    void print_info() const;
private:
    std::shared_ptr<byte[]> pdata;
    unsigned int data_len;
    unsigned int offset;
    vector<int> shape;
    vector<int> strides;
};