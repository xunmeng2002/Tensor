#pragma once
#include <vector>
#include <memory>
#include <iostream>


using namespace std;

template<typename T>
class Tensor
{
public:
	Tensor() = default;
	Tensor(const vector<int>& shape)
		:shape(shape)
	{
		strides = Strides();
	}

	virtual shared_ptr<T> Allocate(size_t elements) const = 0;
	const vector<int>& Shape() const { return shape; }
	shared_ptr<T> Data() { return data; }
	shared_ptr<T> Data() const { return data; }
	virtual shared_ptr<Tensor<T>> Add(const Tensor& other) const = 0;
	size_t Size() const
	{
		size_t size = 1;
		for (int dim : shape)
			size *= dim;
		return size;
	}
	vector<int> Strides() const
	{
		vector<int> strides(shape.size());
		int stride = 1;
		for (int i = (int)shape.size() - 1; i >= 0; --i)
		{
			strides[i] = stride;
			stride *= shape[i];
		}
		return strides;
	}

	bool IsContiguous() const { return true; }
	virtual shared_ptr<Tensor<T>> Contiguous() const = 0;

	void Print() const
	{
		auto dataPtr = data.get();
		for (size_t i = 0; i < Size(); ++i)
			cout << dataPtr[i] << " ";
		cout << endl;
	}

protected:
	shared_ptr<T> data;
	vector<int> shape;
	vector<int> strides; // ”√”⁄¥Ê¥¢Ã›∂»
};




