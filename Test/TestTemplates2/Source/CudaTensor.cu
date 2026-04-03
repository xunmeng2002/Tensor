#include "CudaTensor.h"
#include <cuda_runtime.h>


template<typename T>
shared_ptr<T> CudaTensor<T>::Allocate(size_t elements) const
{
	T* deviceData;
	cudaMalloc(&deviceData, elements * sizeof(T));
	return shared_ptr<T>(deviceData, [](T* ptr) { cudaFree(ptr); });
}
template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::Create(const vector<int>& shape)
{
	return make_shared<CudaTensor<T>>(shape);
}

template<typename T>
shared_ptr<CudaTensor<T>> CudaTensor<T>::Arrange(const vector<int>& shape)
{
	auto tensor = make_shared<CudaTensor<T>>(shape);
	auto data = tensor->data.get();
	for (size_t i = 0; i < tensor->Size(); ++i)
		data[i] = static_cast<T>(i);
	return tensor;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Add(const Tensor<T>& other) const
{
	if (this->Shape() != other.Shape())
		return nullptr;
	auto result = make_shared<CudaTensor<T>>(this->Shape());
	auto resultData = result->data.get();
	auto thisData = this->data.get();
	auto otherData = other.Data().get();
	for (size_t i = 0; i < this->Size(); ++i)
		resultData[i] = thisData[i] + otherData[i];
	return result;
}

template<typename T>
shared_ptr<Tensor<T>> CudaTensor<T>::Contiguous() const
{
	auto result = Create(this->shape);
	if (this->IsContiguous())
	{
		*result = *this;
		return result;
	}

	result->data = Allocate(this->Size());
	result->shape = this->shape;
	result->strides = result->Strides();
	return result;
}



template class CudaTensor<float>;
template class CudaTensor<double>;
template class CudaTensor<int>;

