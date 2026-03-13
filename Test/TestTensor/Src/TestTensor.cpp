#include "TestTensor.h"

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
void PrintRowWithIndices(int* row_start, int inner_size, int inner_stride, const vector<int>& out_indices)
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

template<typename T>
void PrintTensor(Tensor<T>* tensor)
{
    PrintDimOrder(tensor->GetDimensionOrderByStride());
    tensor->PrintInfo();
    tensor->ForEachBatch(PrintRow);
    tensor->ForEachBatchWithIndices(PrintRowWithIndices);
    printf("\n");
}
Tensor<int>* PrepareTestTensor()
{
    int* array = new int[24] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    vector<int> shape = { 2, 4, 3 };
    vector<int> strides = { 12, 3, 1 };
    auto tensor = new Tensor(array, 24, 0, shape, strides);
    PrintTensor(tensor);
    return tensor;
}
Tensor<int>* PrepareTestTensor2()
{
    int* array = new int[12] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    vector<int> shape = { 2, 2, 3 };
    vector<int> strides = { 6, 3, 1 };
    auto tensor = new Tensor(array, 12, 0, shape, strides);
    PrintTensor(tensor);
    return tensor;
}
void TestPermuteAndTranspose()
{
    printf("\n--------Start TestPermuteAndTranspose--------\n");
    auto tensor = PrepareTestTensor();
    
    printf("Transpose for dim 2 and 1\n");
    tensor->Transpose(2, 1);
    PrintTensor(tensor);

    printf("Permute for {0, 2, 1}\n");
    tensor->Permute({ 0, 2, 1 });
    PrintTensor(tensor);
    printf("--------End TestPermuteAndTranspose--------\n\n");
}
void TestContiguous()
{
    printf("\n--------Start TestContiguous--------\n");
    auto tensor = PrepareTestTensor();

    printf("Transpose for dim 2 and 1\n");
    tensor->Transpose(2, 1);
    PrintTensor(tensor);

    auto tensor2 = tensor->Contiguous();
    printf("Contiguous\n");
    PrintTensor(tensor2);
    tensor2->ForEachLinear(Print);

    tensor->ContiguousInplace();
    printf("ContiguousInplace\n");
    PrintTensor(tensor);
    tensor->ForEachLinear(Print);

    printf("--------End TestContiguous--------\n\n");
}
void TestView()
{
    printf("\n--------Start TestView--------\n");
    auto tensor = PrepareTestTensor();

    auto tensor2 = tensor->View({ 3,8 });
    PrintTensor(tensor2);

    printf("Transpose for dim 0 and 1\n");
    tensor2->Transpose(0, 1);
    auto tensor3 = tensor2->View({ 6, 4 });
    if (tensor3 == nullptr)
    {
        printf("Create View From tensor2 Failed\n");
    }
    else
    {
        PrintTensor(tensor3);
    }

    printf("--------End TestView--------\n\n");
}
void TestSqueezeAndUnsqueeze()
{
    printf("\n--------Start TestSqueezeAndUnsqueeze--------\n");
    auto tensor = PrepareTestTensor();
    printf("Unsqueeze for dim 1\n");
    auto tensor2 = tensor->Unsqueeze(1);
    PrintTensor(tensor2);
    printf("Squeeze for dim 1\n");
    auto tensor3 = tensor2->Squeeze(1);
    PrintTensor(tensor3);

    printf("Squeeze\n");
    auto tensor4 = tensor2->Squeeze();
    PrintTensor(tensor4);
	printf("--------End TestSqueezeAndUnsqueeze--------\n\n");
}
void TestExpand()
{
    printf("\n--------Start TestExpand--------\n");
    auto tensor = PrepareTestTensor2();

    printf("Unsqueeze for dim 1\n");
    auto tensor2 = tensor->Unsqueeze(1);
    PrintTensor(tensor2);

    printf("Expand for {2, 2, 2, 3} from {2, 1, 2, 3}\n");
    auto tensor3 = tensor2->Expand({ 2, 2, 2, 3 });
    if (tensor3 == nullptr)
    {
        printf("Expand Failed.\n");
    }
    else
    {
        PrintTensor(tensor3);
    }


    printf("Expand for {3, 2, 2, 3} from {2, 2, 3}\n");
    auto tensor4 = tensor->Expand({ 3, 2, 2, 3 });
    if (tensor4 == nullptr)
    {
        printf("Expand Failed.\n");
    }
    else
    {
        PrintTensor(tensor4);
    }
    printf("--------End TestExpand--------\n\n");
}
void TestRepeat()
{
    printf("\n--------Start TestRepeat--------\n");
    auto tensor = PrepareTestTensor2();

    printf("Repeat for {1, 2, 1}\n");
    auto tensor2 = tensor->Repeat({ 1, 2, 1});
    PrintTensor(tensor2);
    printf("--------End TestRepeat--------\n\n");
}