#pragma once
#include "Tensor.h"

template<typename T>
class CpuTensor : public Tensor<T>
{
public:
	CpuTensor() = default;
	CpuTensor(const vector<int>& shape)
		: Tensor<T>(shape)
	{
		this->data = this->Allocate(this->Size());
	}

	virtual shared_ptr<T> Allocate(size_t elements) const override;
	static shared_ptr<CpuTensor<T>> Create(const vector<int>& shape);
	static shared_ptr<Tensor<T>> Arrange(const vector<int>& shape);
	virtual shared_ptr<Tensor<T>> Add(const Tensor<T>& other) const override;

	virtual shared_ptr<Tensor<T>> Contiguous() const override;
};




