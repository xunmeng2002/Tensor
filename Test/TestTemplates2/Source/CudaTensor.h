#pragma once
#include "Tensor.h"

template<typename T>
class CudaTensor : public Tensor<T>
{
public:
	CudaTensor() = default;
	CudaTensor(const vector<int>& shape)
		: Tensor<T>(shape)
	{
		this->data = this->Allocate(this->Size());
	}
	
	virtual shared_ptr<T> Allocate(size_t elements) const override;
	static shared_ptr<CudaTensor<T>> Create(const vector<int>& shape);
	static shared_ptr<CudaTensor<T>> Arrange(const vector<int>& shape);
	virtual shared_ptr<Tensor<T>> Add(const Tensor<T>& other) const override;
	virtual shared_ptr<Tensor<T>> Contiguous() const override;
};




