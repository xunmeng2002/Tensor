#pragma once
#include "CpuTensor.h"
#include "CudaTensor.h"
#include "CudaTensorKernel.cuh"


template<typename Src, typename Dest>
shared_ptr<CpuTensor<Dest>> CpuTensorConvert(const CpuTensor<Src>* srcTensor)
{
	auto elements = srcTensor->GetTotalElements();
	auto destTensor = CpuTensor<Dest>::CreateCpuTensor(srcTensor->GetShape());
	auto srcData = srcTensor->GetData().get();
	auto destData = destTensor->GetData().get();
	if (srcTensor->Contiguous())
	{
		for (auto i = 0; i < elements; ++i)
		{
			destData[i] = (Dest)srcData[i];
		}
	}
	else
	{
		vector<int> indices(srcTensor->GetShape().size());
		for (auto i = 0; i < elements; ++i)
		{
			auto srcOffset = srcTensor->CalculateOffset(indices);
			destData[i] = (Dest)srcData[srcOffset];
			srcTensor->UpdateIndices(indices);
		}
	}
	return destTensor;
}
template<typename Src, typename Dest>
shared_ptr<CudaTensor<Dest>> CudaTensorConvert(const CudaTensor<Src>* srcTensor)
{
	auto dims = srcTensor->GetDim();
	auto elements = srcTensor->GetTotalElements();
	auto destTensor = CudaTensor<Dest>::CreateCudaTensor(srcTensor->GetShape());

	auto srcData = srcTensor->GetData().get() + srcTensor->GetOffset();
	auto destData = destTensor->GetData().get();
	auto dShape = CopyVectorToDevice(srcTensor->GetShape());
	auto dStrides = CopyVectorToDevice(srcTensor->GetStrides());
	auto gridSize = srcTensor->GetDefaultGridSize(elements);

	if (srcTensor->Contiguous())
	{
		CopyDataToDestTypeContiguousKernel<<<gridSize, DefaultBlockSize>>>(srcData, destData, (int*)dShape.get(), (int*)dStrides.get(), dims, elements);
	}
	else
	{
		CopyDataToDestTypeNoncontiguousKernel<<<gridSize, DefaultBlockSize>>> (srcData, destData, (int*)dShape.get(), (int*)dStrides.get(), dims, elements);
	}
	return destTensor;
}

