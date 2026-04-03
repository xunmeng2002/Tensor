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
    virtual std::shared_ptr<T> Allocate(size_t elements) const override;


    // 张量操作
    virtual shared_ptr<Tensor<T>> Contiguous() const override;
    virtual void Contiguous_() override;
    virtual shared_ptr<T> CopyDataToHost() const override;
    virtual shared_ptr<Tensor<T>> Create() const override;
    virtual shared_ptr<Tensor<T>> Create(const std::vector<int>& shape) const override;
    virtual bool HasZero() const override;
    virtual void RepeatRecursive(int dim, const std::vector<int>& repeats, T*& destData, vector<int>& indices) const override;

    // 迭代器支持（用于兼容性，但CUDA张量通常不使用迭代器）
    class iterator;
    class const_iterator;

    // 工厂方法
    static shared_ptr<CudaTensor<T>> CreateCudaTensor();
    static shared_ptr<CudaTensor<T>> CreateCudaTensor(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> Full(const std::vector<int>& shape, T value);
    static shared_ptr<CudaTensor<T>> Ones(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> Zeros(const std::vector<int>& shape);
    static shared_ptr<CudaTensor<T>> Arrange(const std::vector<int>& shape, T start, T step);


protected:
    virtual void CopyDataContiguous(T* dest) const override;
    virtual void CopyDataNoncontiguous(T* dest) const override;
    std::shared_ptr<char> CopyShapeToDevice() const;
    std::shared_ptr<char> CopyStridesToDevice() const;

    void CopyRecursive(int dim, T*& dest, size_t currOffset) const;
    void FillValue(T value);
    void FillValueWithStep(T start, T step);
    int GetDefaultGridSize(size_t elements) const;

    // 辅助方法
    void CheckAndBroadcast(const Tensor<T>& other, std::vector<int>& broadcastShape) const;
    shared_ptr<Tensor<T>> BroadcastOp(const Tensor<T>& other, const std::string& op) const;

protected:
    int defaultBlockSize = 256;
};


