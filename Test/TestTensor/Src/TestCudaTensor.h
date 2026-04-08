#pragma once
#include "Tensor.h"
#include "CudaTensor.h"


class TestCudaTensor
{
public:
	static void Test();


protected:
	static void TestPermuteAndTranspose();
	static void TestContiguous();
	static void TestView();
	static void TestSqueezeAndUnsqueeze();
	static void TestExpand();
	static void TestRepeat();
	static void TestNarrow();
	static void TestZerosAndOnes();


	static void TestNarrowAndClone();
	static void TestTransposeAndClone();
	static void TestNarrowTransposeAndClone();
	static void TestExpandAndClone();
};
