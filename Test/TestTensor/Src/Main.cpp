#include "Tensor.h"
#include <vector>

using namespace std;

void Print(const int& value, const vector<int>& indices)
{
    for (auto i = 0; i < indices.size(); ++i)
    {
        printf("[%d]", indices[i]);
    }
    printf(" = %d\n", value);
}
void PrintRow(int* row_start, int inner_size, int inner_stride)
{
    printf("[");
    for (auto i = 0; i < inner_size; ++i)
    {
        printf("%d, ", *row_start);
        row_start += inner_stride;
    }
    printf("]\n");
}
void PrintRow2(int* row_start, int inner_size, int inner_stride, const vector<int>& out_indices)
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
void PrintDimOrder(const vector<int>& values)
{
    printf("DimOrder:[");
    for (auto& value : values)
    {
        printf("%d, ", value);
    }
    printf("]\n");
}


int main()
{
    int* array = new int[24] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    vector<int> shape = { 2, 4, 3 };
    vector<int> strides = { 12, 3, 1 };
    Tensor tensor((byte*)array, 24, 0, shape, strides);
    tensor.for_each_batch(PrintRow);
    tensor.for_each_batch_with_indices(PrintRow2);
    tensor.print_info();
    PrintDimOrder(tensor.get_dimension_order_by_stride());


    tensor.permute({ 0, 2, 1 });
    //tensor.transpose(2, 1);
    tensor.for_each_batch(PrintRow);
    tensor.for_each_batch_with_indices(PrintRow2);
    tensor.print_info();
    PrintDimOrder(tensor.get_dimension_order_by_stride());

    auto tensor2 = tensor.contiguous();
    tensor2.for_each_batch(PrintRow);
    tensor2.for_each_batch_with_indices(PrintRow2);
    tensor2.print_info();
    PrintDimOrder(tensor2.get_dimension_order_by_stride());
    tensor2.for_each_linear(Print);


    //tensor.permute({ 0, 2, 1 });
    //tensor.transpose(2, 1);
    //tensor.for_each_batch(PrintRow);
    //tensor.for_each_batch_with_indices(PrintRow2);
    //tensor.print_info();
    //PrintDimOrder(tensor.get_dimension_order_by_stride());

    return 0;
}
