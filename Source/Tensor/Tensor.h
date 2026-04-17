#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <memory>
#include <numeric>

using namespace std;

enum class DeviceType : int { CPU, CUDA };
// 运算类型枚举
enum class ElementwiseOp { ADD = 0, SUB, MUL, DIV };
enum class ScalarOp { ADD = 0, SUB, MUL, DIV };

template<typename T>
class Tensor
{
public:
    using ValueType = T;
    Tensor(DeviceType device) noexcept;
    Tensor(const vector<int>& shape, DeviceType device) noexcept;
    Tensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape, DeviceType device) noexcept;
    Tensor(T* data, std::function<void(void*)> deleter, uint64_t offset, const vector<int>& shape, const vector<int>& strides, DeviceType device) noexcept;
    Tensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, DeviceType device) noexcept;
    Tensor(shared_ptr<T> data, uint64_t offset, const vector<int>& shape, const vector<int>& strides, DeviceType device) noexcept;
    virtual ~Tensor() = default;

    Tensor(const Tensor& other) noexcept;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    virtual T operator[](int64_t index) const = 0;
    virtual T operator[](const std::vector<int>& indices) const = 0;
    template<typename... Indices>
    T& operator()(Indices... indices);
    
    shared_ptr<Tensor> operator+(const Tensor& other) const;
    shared_ptr<Tensor> operator-(const Tensor& other) const;
    shared_ptr<Tensor> operator*(const Tensor& other) const;
    shared_ptr<Tensor> operator/(const Tensor& other) const;
    shared_ptr<Tensor> operator+(T value) const;
    shared_ptr<Tensor> operator-(T value) const;
    shared_ptr<Tensor> operator*(T value) const;
    shared_ptr<Tensor> operator/(T value) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    Tensor& operator+=(T value);
    Tensor& operator-=(T value);
    Tensor& operator*=(T value);
    Tensor& operator/=(T value);

    virtual shared_ptr<Tensor<T>> Add(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Sub(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Mul(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Div(const Tensor<T>& other) const = 0;
    virtual shared_ptr<Tensor<T>> Add(T value) const = 0;
    virtual shared_ptr<Tensor<T>> Sub(T value) const = 0;
    virtual shared_ptr<Tensor<T>> Mul(T value) const = 0;
    virtual shared_ptr<Tensor<T>> Div(T value) const = 0;
    virtual void Add_(const Tensor<T>& other) = 0;
    virtual void Sub_(const Tensor<T>& other) = 0;
    virtual void Mul_(const Tensor<T>& other) = 0;
    virtual void Div_(const Tensor<T>& other) = 0;
    virtual void Add_(T value) = 0;
    virtual void Sub_(T value) = 0;
    virtual void Mul_(T value) = 0;
    virtual void Div_(T value) = 0;
    virtual T At(const std::vector<int>& indices) const = 0;
    virtual void SetAt(const std::vector<int>& indices, const T& value) = 0;


    uint64_t CalculateOffset(const std::vector<int>& indices) const;
    static uint64_t CalculateOffset(const std::vector<int>& indices, uint64_t offset, const std::vector<int>& shape, const std::vector<int>& strides);
    uint64_t CalculateOffsetUnChecked(const std::vector<int>& indices) const;
    bool CheckDims(const vector<int>& dims);
    virtual bool CheckEqual(const vector<T>& data) const = 0;
    shared_ptr<Tensor> Clone() const;
    virtual shared_ptr<Tensor<T>> Contiguous() const = 0;
    virtual void Contiguous_() = 0;
    virtual shared_ptr<Tensor<T>> Create() const = 0;
    virtual shared_ptr<Tensor<T>> Create(const std::vector<int>& shape) const = 0;
    shared_ptr<Tensor> Expand(const std::vector<int>& newShape) const;
    std::vector<int> GetContiguousStrides() const;
    static std::vector<int> GetContiguousStrides(const std::vector<int>& shape);
    shared_ptr<T> GetData() const;
    DeviceType GetDevice() const;
    int GetDim() const;
    size_t GetOffset() const;
    const std::vector<int> GetShape() const;
    const std::vector<int> GetStrides() const;
    size_t GetTotalElements() const;
    static size_t GetTotalElements(const std::vector<int>& shape);
    bool HasValidDenseLayout() const;
    virtual bool HasZero() const = 0;
    bool IsContiguous() const;
    shared_ptr<Tensor> Narrow(const std::vector<int>& dims, const std::vector<int>& starts, const std::vector<int>& lengths) const;
    shared_ptr<Tensor> Narrow(int dim, int start, int length) const;
    bool Permute(const vector<int>& dims);
    void Print(bool pretty = true) const;
    void PrintInfo() const;
    shared_ptr<Tensor> Repeat(int dim, int repeat) const;
    shared_ptr<Tensor> Repeat(const std::vector<int>& repeats) const;
    shared_ptr<Tensor> Reshape(const std::vector<int>& newShape) const;
    shared_ptr<Tensor> Squeeze() const;
    shared_ptr<Tensor> Squeeze(int dim) const;
    shared_ptr<Tensor> Squeeze(const std::vector<int>& dims) const;
	virtual shared_ptr<Tensor> ToDevice(DeviceType targetDevice) const = 0;
    bool Transpose(int dim1, int dim2);
    shared_ptr<Tensor> Unsqueeze(int dim) const;
    void UpdateIndices(std::vector<int>& indices) const;
    shared_ptr<Tensor> View(const std::vector<int>& newShape) const;

protected:
    bool CheckDevice(const Tensor& other) const;
    bool CheckShape(const Tensor& other) const;
    bool CheckIndex(const std::vector<int>& indices) const;
    shared_ptr<Tensor> CloneContiguous() const;
    shared_ptr<Tensor> CloneDenseView() const;
    virtual void CopyDataContiguous(T* dest) const = 0;
    virtual void CopyDataNoncontiguous(T* dest) const = 0;
    virtual void CopyDataRepeat(Tensor<T>* repeatTensor) const = 0;
    std::vector<int> LinearToIndices(size_t linearIndex) const;
    static void PrintCompactRecursive(int dim, T* data, const std::vector<int>& shape, const std::vector<int>& strides);
    static void PrintPrettyRecursive(int dim, T* data, const std::vector<int>& shape, const std::vector<int>& strides);
    bool ValidShape(vector<int>& newShape) const;
    shared_ptr<Tensor> ViewImpl(const std::vector<int>& newShape) const;

protected:
    std::shared_ptr<T> pdata;
    uint64_t offset;
    vector<int> shape;
    vector<int> strides;
    DeviceType device;
};


extern template class Tensor<float>;
extern template class Tensor<double>;
extern template class Tensor<int8_t>;
extern template class Tensor<uint8_t>;
extern template class Tensor<int16_t>;
extern template class Tensor<uint16_t>;
extern template class Tensor<int32_t>;
extern template class Tensor<uint32_t>;
extern template class Tensor<int64_t>;
extern template class Tensor<uint64_t>;


