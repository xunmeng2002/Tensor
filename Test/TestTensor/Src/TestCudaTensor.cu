#include "TestCudaTensor.h"
#include "CudaTensor.h"
#include "CudaTensorKernel.cuh"
#include <exception>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;



template<typename TensorType>
static void CheckTensor(TensorType* tensor, size_t expectOffset, const vector<int>& expectShape, const vector<int>& expectStrides,
    bool expectContiguous, const vector<typename TensorType::ValueType>& expectElements)
{
    using T = typename TensorType::ValueType;
	assert(tensor->GetOffset() == expectOffset);
	assert(tensor->GetShape() == expectShape);
	assert(tensor->GetStrides() == expectStrides);
	assert(tensor->IsContiguous() == expectContiguous);
    size_t total = tensor->GetTotalElements();
    assert(total == expectElements.size());
    assert(tensor->CheckEqual(expectElements));
}

template<typename TensorType>
static void PrintTensor(TensorType* tensor)
{
    tensor->PrintInfo();
    tensor->Print();
    printf("\n");
}
// Prepare a 2x3x4 tensor with values from 1 to 24, and strides {12, 4, 1}
static shared_ptr<CudaTensor<int>> PrepareTestTensor()
{
    vector<int> shape = { 2, 3, 4 };
    auto tensor = CudaTensor<int>::Arrange(shape, 1);
    PrintTensor(tensor.get());
    return tensor;
}
// Prepare a 2x2x3 tensor with values from 1 to 12, and strides {6, 3, 1}
static shared_ptr<CudaTensor<int>> PrepareTestTensor2()
{
    vector<int> shape = { 2, 2, 3 };
    auto tensor = CudaTensor<int>::Arrange(shape, 1);
    PrintTensor(tensor.get());
    return tensor;
}

void TestCudaTensor::Test()
{
    TestPermuteAndTranspose();
    TestContiguous();
    TestView();
    TestSqueezeAndUnsqueeze();
    TestExpand();
    TestRepeat();
    TestNarrow();
    TestZerosAndOnes();

    TestNarrowAndClone();
    TestTransposeAndClone();
    TestNarrowTransposeAndClone();
    TestExpandAndClone();
}
void TestCudaTensor::TestPermuteAndTranspose()
{
    printf("\n--------Start TestPermuteAndTranspose--------\n");
    auto tensor1 = PrepareTestTensor();
    
    printf("Transpose for dim 0 and 1\n");
    tensor1->Transpose(0, 1);
    PrintTensor(tensor1.get());
    CheckTensor(tensor1.get(), 0, {3, 2, 4}, {4, 12, 1}, false,
        { 1, 2, 3, 4, 13, 14, 15, 16, 5, 6, 7, 8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24 });
    

    printf("Permute for {0, 2, 1}\n");
    tensor1->Permute({ 0, 2, 1 });
    PrintTensor(tensor1.get());
    CheckTensor(tensor1.get(), 0, { 3, 4, 2 }, { 4, 1, 12 }, false,
        { 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24 });
    printf("--------End TestPermuteAndTranspose--------\n\n");
}
void TestCudaTensor::TestContiguous()
{
    printf("\n--------Start TestContiguous--------\n");
    auto tensor1 = PrepareTestTensor();

    printf("Transpose for dim 2 and 1\n");
    tensor1->Transpose(2, 1);
    PrintTensor(tensor1.get());
    CheckTensor(tensor1.get(), 0, { 2, 4, 3 }, { 12, 1, 4 }, false,
		{ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24 });

    printf("Contiguous\n");
    auto tensor2 = tensor1->Contiguous();
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, { 2, 4, 3 }, { 12, 3, 1 }, true,
        { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24 });
    //tensor2->ForEachLinear(Print);

    tensor1->Contiguous_();
    printf("Contiguous_\n");
    PrintTensor(tensor1.get());
    CheckTensor(tensor1.get(), 0, { 2, 4, 3 }, { 12, 3, 1 }, true,
        { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24 });
    //tensor->ForEachLinear(Print);

    printf("--------End TestContiguous--------\n\n");
}
void TestCudaTensor::TestView()
{
    printf("\n--------Start TestView--------\n");
    auto tensor = PrepareTestTensor();

    printf("View for {3, 8}\n");
    auto tensor2 = tensor->View({ 3,8 });
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, { 3, 8 }, { 8, 1 }, true,
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });

    printf("Transpose for dim 0 and 1\n");
    tensor2->Transpose(0, 1);
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, { 8, 3 }, { 1, 8 }, false,
		{ 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5, 13, 21, 6, 14, 22, 7, 15, 23, 8, 16, 24 });

    auto tensor3 = tensor2->View({ 6, 4 });
    assert(tensor3 == nullptr);
    if (tensor3 == nullptr)
    {
        printf("Create View From tensor2 Failed\n");
    }

    printf("--------End TestView--------\n\n");
}
void TestCudaTensor::TestSqueezeAndUnsqueeze()
{
    printf("\n--------Start TestSqueezeAndUnsqueeze--------\n");
    auto tensor1 = PrepareTestTensor();
    
    printf("Unsqueeze for dim 1\n");
    auto tensor2 = tensor1->Unsqueeze(1);
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, {2, 1, 3, 4}, {12, 12, 4, 1}, true,
        { 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });

    printf("Squeeze for dim 1\n");
    auto tensor3 = tensor2->Squeeze(1);
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 0, { 2, 3, 4 }, { 12, 4, 1 }, true,
        { 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });

    printf("Squeeze\n");
    auto tensor4 = tensor2->Squeeze();
    PrintTensor(tensor4.get());
    CheckTensor(tensor3.get(), 0, { 2, 3, 4 }, { 12, 4, 1 }, true,
        { 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });

	printf("--------End TestSqueezeAndUnsqueeze--------\n\n");
}
void TestCudaTensor::TestExpand()
{
    printf("\n--------Start TestExpand--------\n");
    auto tensor1 = PrepareTestTensor2();

    printf("Unsqueeze for dim 1\n");
    auto tensor2 = tensor1->Unsqueeze(1);
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, { 2, 1, 2, 3 }, { 6, 6, 3, 1 }, true,
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    printf("Expand for {2, 2, 2, 3} from {2, 1, 2, 3}\n");
    auto tensor3 = tensor2->Expand({ 2, 2, 2, 3 });
    assert(tensor3 != nullptr);
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 0, { 2, 2, 2, 3 }, { 6, 0, 3, 1 }, false,
        {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12});

    printf("Expand for {3, 2, 2, 3} from {2, 2, 3}\n");
    auto tensor4 = tensor1->Expand({ 3, 2, 2, 3 });
    assert(tensor4 != nullptr);
    PrintTensor(tensor4.get());
    CheckTensor(tensor4.get(), 0, { 3, 2, 2, 3 }, { 0, 6, 3, 1 }, false,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    printf("--------End TestExpand--------\n\n");
}
void TestCudaTensor::TestRepeat()
{
    printf("\n--------Start TestRepeat--------\n");
    auto tensor1 = PrepareTestTensor2();

    printf("Repeat for {1, 2, 1}\n");
    auto tensor2 = tensor1->Repeat({ 1, 2, 1});
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, { 2, 4, 3 }, { 12, 3, 1 }, true,
        { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12 });

    printf("--------End TestRepeat--------\n\n");
}
void TestCudaTensor::TestNarrow()
{
    printf("\n--------Start TestNarrow--------\n");
    auto tensor1 = PrepareTestTensor();
    
    printf("Narrow for dim 1, start 1, length 2, from {2, 3, 4}\n");
    auto tensor2 = tensor1->Narrow(1, 1, 2);
    assert(tensor2 != nullptr);
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 4, { 2, 2, 4 }, { 12, 4, 1 }, false,
        { 5, 6, 7, 8, 9, 10, 11,12, 17, 18, 19, 20, 21, 22, 23, 24 });

    printf("Narrow for dim 0, start 0, length 1 from {2, 2, 4}\n");
	auto tensor3 = tensor2->Narrow(0, 0, 1);
    assert(tensor3 != nullptr);
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 4, { 1, 2, 4 }, { 12, 4, 1 }, true,
        { 5, 6, 7, 8, 9, 10, 11,12 });

    printf("Narrow for dim {1, 0}, start {1, 0}, length {2, 1}, from {2, 3, 4}\n");
    auto tensor4 = tensor1->Narrow({ 1,0 }, { 1, 0 }, {2, 1});
    assert(tensor4 != nullptr);
    PrintTensor(tensor4.get());
    CheckTensor(tensor4.get(), 4, { 1, 2, 4 }, { 12, 4, 1 }, true,
        { 5, 6, 7, 8, 9, 10, 11,12 });


	printf("--------End TestNarrow--------\n\n");
}
void TestCudaTensor::TestZerosAndOnes()
{
    printf("\n--------Start TestZerosAndOnes--------\n");
    printf("Zeros for {2, 3, 4}\n");
    auto tensor1 = CudaTensor<int>::Zeros({ 2, 3, 4 });
    PrintTensor(tensor1.get());
    CheckTensor(tensor1.get(), 0, { 2, 3, 4 }, { 12, 4, 1 }, true,
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    
    printf("Ones for {2, 3, 4}\n");
    auto tensor2 = CudaTensor<int>::Ones({ 2, 3, 4 });
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, { 2, 3, 4 }, { 12, 4, 1 }, true,
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    
    printf("Full with value 5 for {2, 3, 4}\n");
    auto tensor3 = CudaTensor<int>::Full({ 2, 3, 4 }, 5);
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 0, { 2, 3, 4 }, { 12, 4, 1 }, true,
        { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 });

	printf("--------End TestZerosAndOnes--------\n\n");
}


void TestCudaTensor::TestNarrowAndClone()
{
    printf("\n--------Start TestNarrowAndClone--------\n");
    auto tensor1 = PrepareTestTensor();

    printf("Narrow for dim 0, start 1, length 1, from {2, 3, 4}\n");
    auto tensor2 = tensor1->Narrow(0, 1, 1);
    assert(tensor2 != nullptr);
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 12, { 1, 3, 4 }, { 12, 4, 1 }, true,
        { 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });

    printf("Clone\n");
    auto tensor3 = tensor2->Clone();
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 0, { 1, 3, 4 }, { 12, 4, 1 }, true,
        { 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });

    printf("Narrow for dim 1, start 0, length 2, from {2, 3, 4}\n");
    auto tensor4 = tensor1->Narrow(1, 0, 2);
    assert(tensor4 != nullptr);
    PrintTensor(tensor4.get());
    CheckTensor(tensor4.get(), 0, { 2, 2, 4 }, { 12, 4, 1 }, false,
        { 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20 });

    printf("Clone\n");
    auto tensor5 = tensor4->Clone();
    PrintTensor(tensor5.get());
    CheckTensor(tensor5.get(), 0, { 2, 2, 4 }, { 8, 4, 1 }, true,
        { 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20 });

    printf("\n--------End TestNarrowAndClone--------\n");
}
void TestCudaTensor::TestTransposeAndClone()
{
    printf("\n--------Start TestTransposeAndClone--------\n");
    auto tensor1 = PrepareTestTensor();

    printf("Transpose for dim 1 and 2\n");
    tensor1->Transpose(1, 2);
    assert(tensor1 != nullptr);
    PrintTensor(tensor1.get());
    CheckTensor(tensor1.get(), 0, { 2, 4, 3 }, { 12, 1, 4 }, false,
        { 1, 5,9,2,6,10,3,7,11,4,8,12,13,17,21,14, 18,22, 15, 19, 23, 16, 20, 24 });

    printf("Clone\n");
    auto tensor3 = tensor1->Clone();
    PrintTensor(tensor3.get());
    CheckTensor(tensor1.get(), 0, { 2, 4, 3 }, { 12, 1, 4 }, false,
        { 1, 5,9,2,6,10,3,7,11,4,8,12,13,17,21,14, 18,22, 15, 19, 23, 16, 20, 24 });

    printf("\n--------End TestTransposeAndClone--------\n");
}
void TestCudaTensor::TestNarrowTransposeAndClone()
{
    printf("\n--------Start TestNarrowTransposeAndClone--------\n");
    auto tensor1 = PrepareTestTensor();

    printf("Narrow for dim 1, start 1, length 2, from {2, 3, 4}\n");
    auto tensor2 = tensor1->Narrow(1, 1, 2);
    assert(tensor2 != nullptr);
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 4, { 2, 2, 4 }, { 12, 4, 1 }, false,
        { 5, 6, 7, 8, 9, 10, 11,12, 17, 18, 19, 20, 21, 22, 23, 24 });

    printf("Narrow for dim 2, start 1, length 2, from {2, 2, 4}\n");
    auto tensor3 = tensor2->Narrow(2, 1, 2);
    assert(tensor3 != nullptr);
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 5, { 2, 2, 2 }, { 12, 4, 1 }, false,
        { 6, 7, 10, 11, 18, 19, 22, 23 });

	printf("Transpose for dim 1 and 2\n");
	tensor3->Transpose(1, 2);
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 5, { 2, 2, 2 }, { 12, 1, 4 }, false,
        { 6, 10, 7, 11, 18, 22, 19, 23 });


    printf("Clone\n");
    auto tensor4 = tensor3->Clone();
    PrintTensor(tensor4.get());
    CheckTensor(tensor4.get(), 0, { 2, 2, 2 }, { 4, 2, 1 }, true,
        { 6, 10, 7, 11, 18, 22, 19, 23 });

    printf("\n--------End TestNarrowTransposeAndClone--------\n");
}
void TestCudaTensor::TestExpandAndClone()
{
    printf("\n--------Start TestExpandAndClone--------\n");
    auto tensor1 = PrepareTestTensor2();


    printf("Expand for {2, 2, 2, 3} from {2, 2, 3}\n");
    auto tensor2 = tensor1->Expand({ 2, 2, 2, 3 });
    assert(tensor2 != nullptr);
    PrintTensor(tensor2.get());
    CheckTensor(tensor2.get(), 0, {2, 2, 2, 3}, {0, 6, 3, 1}, false,
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    printf("Clone\n");
    auto tensor3 = tensor2->Clone();
    assert(tensor3 != nullptr);
    PrintTensor(tensor3.get());
    CheckTensor(tensor3.get(), 0, {2, 2, 2, 3}, {12, 6, 3, 1}, true,
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    printf("--------End TestExpandAndClone--------\n\n");
}



