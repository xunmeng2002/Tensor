#pragma once
#include "Tensor.h"
#include <functional>
#include <memory>
#include <vector>


template<typename T>
class CpuTensor : public Tensor<T>
{
public:
    using ValueType = T;
    CpuTensor() noexcept;
    CpuTensor(const std::vector<int>& shape) noexcept;
    CpuTensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape) noexcept;
    CpuTensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape, const vector<int>& strides) noexcept;
    CpuTensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape) noexcept;
    CpuTensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, const vector<int>& strides) noexcept;
    virtual ~CpuTensor() = default;
    virtual T operator[](int64_t index) const override;
    virtual T operator[](const std::vector<int>& indices) const override;
    virtual T& operator[](int64_t index);
    virtual T& operator[](const std::vector<int>& indices);

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
    static shared_ptr<CpuTensor<T>> CreateCpuTensor();
    static shared_ptr<CpuTensor<T>> CreateCpuTensor(const std::vector<int>& shape);
    static shared_ptr<CpuTensor<T>> Arrange(const std::vector<int>& shape, T start, T step = 1);
    static shared_ptr<CpuTensor<T>> Full(const std::vector<int>& shape, T value);
    static shared_ptr<CpuTensor<T>> Ones(const std::vector<int>& shape);
    static shared_ptr<CpuTensor<T>> Zeros(const std::vector<int>& shape);
    

    static std::shared_ptr<T> Allocate(size_t elements);
    virtual T At(const std::vector<int>& indices) const override;
    virtual T& At(const std::vector<int>& indices);
    virtual void SetAt(const std::vector<int>& indices, const T& value) override;
    virtual bool CheckEqual(const vector<T>& data) const override;
    virtual shared_ptr<Tensor<T>> Contiguous() const override;
    virtual void Contiguous_() override;
    virtual bool HasZero() const override;
    virtual shared_ptr<Tensor<T>> ToDevice(DeviceType targetDevice) const override;


protected:
    virtual void CopyDataContiguous(T* dest) const override;
    virtual void CopyDataNoncontiguous(T* dest) const override;
    virtual void CopyDataRepeat(Tensor<T>* repeatTensor) const;
    
private:
    void CopyDataRecursive(int dim, T*& dest, size_t currOffset) const;
    void CopyDataRepeatRecursive(int dim, const std::vector<int>& repeats, T*& destData, size_t currOffset) const;
    void FillValue(T value);
    void FillValueWithStep(T start, T step);


    shared_ptr<Tensor<T>> ExecuteElementWise(const Tensor<T>& other, const function<T(T, T)>& op) const;
    void ExecuteElementWiseInplace(const Tensor<T>& other, const function<T(T&, T)>& op);
    shared_ptr<Tensor<T>> ExecuteScalar(const T& value, const function<T(T, T)>& op) const;
    void ExecuteScalarInplace(const T& value, const function<T(T&, T)>& op);
};

extern template class CpuTensor<float>;
extern template class CpuTensor<double>;
extern template class CpuTensor<int8_t>;
extern template class CpuTensor<uint8_t>;
extern template class CpuTensor<int16_t>;
extern template class CpuTensor<uint16_t>;
extern template class CpuTensor<int32_t>;
extern template class CpuTensor<uint32_t>;
extern template class CpuTensor<int64_t>;
extern template class CpuTensor<uint64_t>;

