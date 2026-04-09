#pragma once
#include "Tensor.h"
#include "CpuTensor.h"

class TestTensor
{
public:
	TestTensor(DeviceType device);

	void Test();


protected:
	shared_ptr<Tensor<int>> PrepareTestTensor();
	shared_ptr<Tensor<int>> PrepareTestTensor2();

	void TestPermuteAndTranspose();
	void TestContiguous();
	void TestView();
	void TestSqueezeAndUnsqueeze();
	void TestExpand();
	void TestRepeat();
	void TestNarrow();
	void TestZerosAndOnes();


	void TestNarrowAndClone();
	void TestTransposeAndClone();
	void TestNarrowTransposeAndClone();
	void TestExpandAndClone();

private:
	DeviceType device;
};