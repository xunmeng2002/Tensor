#include "CpuTensor.h"


template<typename T>
shared_ptr<T> CpuTensor<T>::Allocate(size_t elements) const
{
	return shared_ptr<T>(new T[elements], default_delete<T[]>());
}
template<typename T>
shared_ptr<CpuTensor<T>> CpuTensor<T>::Create(const vector<int>& shape)
{
	return make_shared<CpuTensor<T>>(shape);
}

template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Arrange(const vector<int>& shape)
{
	auto tensor = Create(shape);
	auto data = tensor->data.get();
	for (size_t i = 0; i < tensor->Size(); ++i)
		data[i] = static_cast<T>(i);
	return tensor;
}

template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Add(const Tensor<T>& other) const
{
	if (this->Shape() != other.Shape())
		return nullptr;
	auto result = Create(this->Shape());
	auto resultData = result->data.get();
	auto thisData = this->data.get();
	auto otherData = other.Data().get();
	for (size_t i = 0; i < this->Size(); ++i)
		resultData[i] = thisData[i] + otherData[i];
	return result;
}

template<typename T>
shared_ptr<Tensor<T>> CpuTensor<T>::Contiguous() const
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


template class CpuTensor<float>;
template class CpuTensor<double>;
template class CpuTensor<int>;

