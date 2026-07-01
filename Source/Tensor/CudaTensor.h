#pragma once
#include "Tensor.h"
#include <functional>
#include <memory>

constexpr int DefaultBlockSize = 256;

template<typename T>
class CudaTensor : public Tensor<T>
{
public:
    using ValueType = T;
    CudaTensor() noexcept;
    CudaTensor(const std::vector<int>& shape) noexcept;
    CudaTensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape) noexcept;
    CudaTensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape, const vector<int>& strides) noexcept;
    CudaTensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape) noexcept;
    CudaTensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, const vector<int>& strides) noexcept;
    virtual ~CudaTensor() = default;
    virtual T operator[](int64_t index) const override;
    virtual T operator[](const std::vector<int>& indices) const override;
    

    virtual shared_ptr<Tensor<T>> Add(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Sub(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Mul(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Div(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Add(T value) const override;
    virtual shared_ptr<Tensor<T>> Sub(T value) const override;
    virtual shared_ptr<Tensor<T>> Mul(T value) const override;
    virtual shared_ptr<Tensor<T>> Div(T value) const override;
    virtual void Add_(const Tensor<T>& other) override;
    virtual void Sub_(const Tensor<T>& other) override;
    virtual void Mul_(const Tensor<T>& other) override;
    virtual void Div_(const Tensor<T>& other) override;
    virtual void Add_(T value) override;
    virtual void Sub_(T value) override;
    virtual void Mul_(T value) override;
    virtual void Div_(T value) override;

    virtual shared_ptr<Tensor<T>> Create() const override;
    virtual shared_ptr<Tensor<T>> Create(const std::vector<int>& shape) const override;
    static shared_ptr<CudaTensor<T>> CreateCudaTensor();
    static shared_ptr<CudaTensor<T>> CreateCudaTensor(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> CreateFromCpuTensor(const Tensor<T>& cpuTensor);
    static shared_ptr<CudaTensor<T>> Arrange(const std::vector<int>& shape, T start, T step = 1);
    static shared_ptr<CudaTensor<T>> Full(const std::vector<int>& shape, T value);
    static shared_ptr<CudaTensor<T>> Ones(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> Zeros(const std::vector<int>& shape);


    static std::shared_ptr<T> Allocate(size_t elements);
    virtual T At(const std::vector<int>& indices) const override;
    virtual void SetAt(const std::vector<int>& indices, const T& value) override;
    virtual bool CheckEqual(const vector<T>& data) const override;
    virtual shared_ptr<Tensor<T>> Contiguous() const override;
    virtual void Contiguous_() override;
    int GetDefaultGridSize(size_t elements) const;
    virtual bool HasZero() const override;
    virtual shared_ptr<Tensor<T>> ToDevice(DeviceType targetDevice) const override;
	shared_ptr<Tensor<T>> ToHost() const;
protected:
    virtual void CopyDataContiguous(T* dest) const override;
    virtual void CopyDataNoncontiguous(T* dest) const override;
    virtual void CopyDataRepeat(Tensor<T>* repeatTensor) const override;

private:
    void FillValue(T value);
    void FillValueWithStep(T start, T step);
    void LaunchElementwiseKernel(const T* right, T* result, ElementwiseOp op) const;
    void LaunchElementwiseKernelInplace(const T* right, ElementwiseOp op);
    void LaunchScalarKernel(T* result, T scalar, ScalarOp op) const;
    void LaunchScalarKernelInplace(T scalar, ScalarOp op);
};


extern template class CudaTensor<float>;
extern template class CudaTensor<double>;
extern template class CudaTensor<int8_t>;
extern template class CudaTensor<uint8_t>;
extern template class CudaTensor<int16_t>;
extern template class CudaTensor<uint16_t>;
extern template class CudaTensor<int32_t>;
extern template class CudaTensor<uint32_t>;
extern template class CudaTensor<int64_t>;
extern template class CudaTensor<uint64_t>;

