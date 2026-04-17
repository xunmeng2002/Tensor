#pragma once
#include "Tensor.h"


class TestTensor
{
public:
	TestTensor(DeviceType device);

	void Test();


protected:
	shared_ptr<Tensor<int>> PrepareTestTensor();
	shared_ptr<Tensor<int>> PrepareTestTensor2();
	shared_ptr<Tensor<int>> PrepareTestTensor3();

	void TestPermuteAndTranspose();
	void TestContiguous();
	void TestView();
	void TestSqueezeAndUnsqueeze();
	void TestExpand();
	void TestRepeat();
	void TestNarrow();
	void TestZerosAndOnes();
	void TestAtAndSetAt();


	void TestNarrowAndClone();
	void TestTransposeAndClone();
	void TestNarrowTransposeAndClone();
	void TestExpandAndClone();

	void TestAddSubMulDiv();
	void TestTensorConvert();

private:
	DeviceType device;
};