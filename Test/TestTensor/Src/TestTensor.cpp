#include "TestTensor.h"

void print(const int& value, const vector<int>& indices)
{
    for (auto i = 0; i < indices.size(); ++i)
    {
        printf("[%d]", indices[i]);
    }
    printf(" = %d\n", value);
}
void print_row(int* row_start, int inner_size, int inner_stride)
{
    printf("[");
    for (auto i = 0; i < inner_size; ++i)
    {
        printf("%d, ", *row_start);
        row_start += inner_stride;
    }
    printf("]\n");
}
void print_row_with_indices(int* row_start, int inner_size, int inner_stride, const vector<int>& out_indices)
{
    printf("[");
    for (auto indice : out_indices)
    {
        printf("%d, ", indice);
    }
    printf("]:[");
    for (auto i = 0; i < inner_size; ++i)
    {
        printf("%d, ", *row_start);
        row_start += inner_stride;
    }
    printf("]\n");
}
void print_dim_order(const vector<int>& values)
{
    printf("DimOrder:[");
    for (auto& value : values)
    {
        printf("%d, ", value);
    }
    printf("]\n");
}

void print_tensor(Tensor& tensor)
{
    print_dim_order(tensor.get_dimension_order_by_stride());
    tensor.print_info();
    tensor.for_each_batch(print_row);
    tensor.for_each_batch_with_indices(print_row_with_indices);
}
Tensor prepare_test_tensor()
{
    int* array = new int[24] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    vector<int> shape = { 2, 4, 3 };
    vector<int> strides = { 12, 3, 1 };
    auto tensor = Tensor((byte*)array, 24, 0, shape, strides);
    print_tensor(tensor);
    return tensor;
}
void test_permute_and_transpose()
{
    printf("\n--------Start test_permute_and_transpose--------\n");
    auto tensor = prepare_test_tensor();
    
    tensor.transpose(2, 1);
    print_tensor(tensor);

    tensor.permute({ 0, 2, 1 });
    print_tensor(tensor);
    printf("--------End test_permute_and_transpose--------\n\n");
}
void test_contiguous()
{
    printf("\n--------Start test_contiguous--------\n");
    auto tensor = prepare_test_tensor();

    tensor.transpose(2, 1);
    print_tensor(tensor);

    auto tensor2 = tensor.contiguous();
    print_tensor(tensor);
    tensor2.for_each_linear(print);

    printf("--------End test_contiguous--------\n\n");
}


