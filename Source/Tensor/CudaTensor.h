#pragma once
#include "Tensor.h"
#include <functional>
#include <memory>


template<typename T>
class CudaTensor : public Tensor<T>
{
public:
    using ValueType = T;
    CudaTensor() noexcept :Tensor<T>(DeviceType::CUDA) {}
    CudaTensor(const std::vector<int>& shape) noexcept
        :Tensor<T>(shape, DeviceType::CUDA)
    {
        this->pdata = Allocate(this->GetTotalElements());
    }
    CudaTensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape) noexcept
        :Tensor<T>(data, deleter, offset, shape, DeviceType::CUDA)
    {
    }
    CudaTensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape) noexcept
        :Tensor<T>(data, offset, shape, DeviceType::CUDA)
    {
    }
    virtual ~CudaTensor() = default;



    // 算术运算
    virtual shared_ptr<Tensor<T>> Add(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Sub(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Mul(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Div(const Tensor<T>& other) const override;
    virtual shared_ptr<Tensor<T>> Add(T value) const override;
    virtual shared_ptr<Tensor<T>> Sub(T value) const override;
    virtual shared_ptr<Tensor<T>> Mul(T value) const override;
    virtual shared_ptr<Tensor<T>> Div(T value) const override;

    // 原地运算
    virtual void Add_(const Tensor<T>& other) override;
    virtual void Sub_(const Tensor<T>& other) override;
    virtual void Mul_(const Tensor<T>& other) override;
    virtual void Div_(const Tensor<T>& other) override;
    virtual void Add_(T value) override;
    virtual void Sub_(T value) override;
    virtual void Mul_(T value) override;
    virtual void Div_(T value) override;

    // 内存管理
    static std::shared_ptr<T> Allocate(size_t elements);
    static  std::shared_ptr<uint8_t> AllocateBytes(size_t bytes);

    virtual T At(const std::vector<int>& indices, bool checkBounds = true) const override;
    virtual T& At(const std::vector<int>& indices, bool checkBounds = true) override;

    // 张量操作
    virtual shared_ptr<Tensor<T>> Contiguous() const override;
    virtual void Contiguous_() override;
    
    virtual bool HasZero() const override;
    virtual bool CheckEqual(const vector<T>& data) const override;

    // 工厂方法
    virtual shared_ptr<Tensor<T>> Create() const override;
    virtual shared_ptr<Tensor<T>> Create(const std::vector<int>& shape) const override;
    static shared_ptr<CudaTensor<T>> CreateCudaTensor();
    static shared_ptr<CudaTensor<T>> CreateCudaTensor(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> Full(const std::vector<int>& shape, T value);
    static shared_ptr<CudaTensor<T>> Ones(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> Zeros(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> Arrange(const std::vector<int>& shape, T start, T step = 1);


protected:
    virtual void CopyDataContiguous(T* dest) const override;
    virtual void CopyDataNoncontiguous(T* dest) const override;
    virtual void CopyDataRepeat(Tensor<T>* repeatTensor, const std::vector<int>& repeats) const override;
    virtual shared_ptr<T> CopyDataToHost() const override;

private:
    template<typename VecT>
    static std::shared_ptr<uint8_t> CopyVectorToDevice(const vector<VecT>& values);
    void FillValue(T value);
    void FillValueWithStep(T start, T step);
    int GetDefaultGridSize(size_t elements) const;

protected:
    int defaultBlockSize = 256;
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

