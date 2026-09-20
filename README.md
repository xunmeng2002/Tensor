# Tensor

[English](README.en.md)

基于 C++20 的 N 维张量库，语义对齐 PyTorch：以 `shape` + `strides` + `offset` 描述布局，视图操作共享底层存储。

同时提供 CPU（`CpuTensor`）与 CUDA（`CudaTensor`）两套后端，由抽象基类 `Tensor<T>` 统一接口并按设备分派。

---

## 特性

- **双后端同构接口**：CPU 与 CUDA 后端继承同一抽象基类，算子签名完全一致，`ToDevice` / `ToHost` 完成跨设备搬运。

- **PyTorch 风格的布局语义**：`Permute`、`Transpose`、`Narrow`、`Squeeze`、`Unsqueeze`、`Expand`、`View` 均返回共享存储的视图，不复制数据；写入视图对原张量可见。

- **显式模板实例化**：支持 10 种数据类型，实现体集中编译，避免模板膨胀导致的编译时间失控。

- **无第三方依赖**：除 CUDA Runtime 外不引入任何外部库。

- **可执行的对照测试**：测试工程随库构建，期望值与 PyTorch 参考脚本 `TestTensor.py` 对齐，便于逐项比对语义。

---

## 环境要求

| 项目 | 要求 |
| ---- | ---- |
| CMake | 3.20 及以上 |
| C++ 标准 | C++20 |
| CUDA Toolkit | 13.0.0 及以上 |
| CUDA 架构 | 75、86（Turing / Ampere） |
| Windows 编译器 | MSVC，启用 `/W4 /bigobj` |
| Linux 编译器 | GCC 13.4（与 CI 流水线一致） |
| 第三方依赖 | 无（仅 CUDA Runtime） |

> **注意**：CUDA Toolkit 为硬性依赖。`Target/Tensor/CMakeLists.txt` 中 `find_package(CUDAToolkit 13.0.0 REQUIRED)` 会直接终止配置，即使只使用 CPU 后端也必须安装。

> **注意**：根 `CMakeLists.txt` 引用了 `${CMAKE_SOURCE_DIR}/../Libs/PersonalLib` 的 include 与 link 目录，但当前没有任何 target 链接该目录，缺少它不影响构建。

---

## 构建

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows 下推荐使用 Visual Studio 生成器（`-A x64` 可显式指定 64 位平台；x64 主机上默认平台即为 x64，省略亦可）：

```shell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
```

也可以直接使用仓库内置的 `CMakeSettings.json` 预设（Visual Studio「打开 CMake 项目」）：

| 预设 | 生成器 | 构建类型 | 环境 |
| ---- | ---- | ---- | ---- |
| `x64-Debug` | Ninja | Debug | MSVC x64 |
| `x64-Release` | Ninja | RelWithDebInfo | MSVC x64 |
| `WSL-GCC-Debug` | Ninja | Debug | WSL Linux x64 |

构建产物：

| 平台 | 库文件 | 测试程序 |
| ---- | ---- | ---- |
| Linux | `build/Target/Tensor/libTensor.so` | `build/Test/TestTensor/TestTensor` |
| Windows | `build/Target/Tensor/Tensor.dll` | `build/Test/TestTensor/TestTensor.exe` |

使用多配置生成器（如 `Visual Studio 17 2022`）时，上述路径中还会插入配置名，例如 `build/Test/TestTensor/Release/TestTensor.exe`。

Windows 下 `TestTensor` 的构建后步骤会把 `Tensor.dll` 复制到可执行文件同目录，无需手动配置 `PATH`。

`Tensor` 在 Windows 上启用 `WINDOWS_EXPORT_ALL_SYMBOLS`，无需书写 `__declspec(dllexport)`。

---

## 运行测试

```shell
./build/Test/TestTensor/TestTensor
```

`Test/TestTensor/Src/Main.cpp` 通过单个变量切换被测设备：

```cpp
DeviceType device = DeviceType::CPU;
//DeviceType device = DeviceType::CUDA;
```

测试覆盖布局操作（转置 / 置换 / 视图 / 挤压 / 扩展 / 重复 / 切片）、连续化、克隆、逐元素算术、跨数据类型转换与设备搬运，每项均断言 `offset`、`shape`、`strides`、`IsContiguous` 与全部元素值。

---

## 持续集成

`.workflow/pipeline.yml` 为 Gitee Go 流水线：在 `nvidia/cuda:12.0.0-devel-ubuntu22.04` 镜像中以 GCC 13.4 执行

```shell
mkdir build && cd build
cmake -G "Unix Makefiles" ../ && make -j
```

该流水线**只构建、不执行测试**（未调用 `./Test/TestTensor`）。

> **注意**：流水线镜像的 CUDA 版本（12.0）低于 `Target/Tensor/CMakeLists.txt` 中 `find_package(CUDAToolkit 13.0.0 REQUIRED)` 的要求，流水线的实际可运行性需要核实。

仓库另有一份 `.gitee-ci.yml`，脚本通过 vcpkg 安装 `opencv` 与 `boost-system` 后构建并运行 `./Test/TestTensor`。这两个库并非本项目的依赖，该文件疑为遗留或复制的模板配置。

---

## 目录结构

```text
Tensor/
├── Source/Tensor/         # 库源码
├── Target/Tensor/         # 库 target 的 CMake 定义
├── Test/TestTensor/       # 测试可执行程序
├── TestTensor.py          # PyTorch 对照脚本
├── ConvertToUtf8Bom.py    # 源码转 UTF-8 BOM 工具（保留原行尾）
├── CMakeSettings.json     # Visual Studio CMake 预设
└── .workflow/pipeline.yml # Gitee Go 流水线定义
```

核心源文件职责：

| 文件 | 职责 |
| ---- | ---- |
| `Tensor.h` / `Tensor.cpp` | 抽象基类、布局推导、视图构造、算子分派 |
| `CpuTensor.h` / `CpuTensor.cpp` | CPU 后端实现与工厂方法 |
| `CudaTensor.h` / `CudaTensor.cu` | CUDA 后端实现与显存管理 |
| `CudaTensorKernel.cuh` / `.cu` | 设备端核函数 |
| `TensorConvert.cuh` | 跨数据类型转换（`CpuTensorConvert` / `CudaTensorConvert`） |

---

## 支持的数据类型

| 类别 | 类型 |
| ---- | ---- |
| 浮点 | `float`、`double` |
| 有符号整型 | `int8_t`、`int16_t`、`int32_t`、`int64_t` |
| 无符号整型 | `uint8_t`、`uint16_t`、`uint32_t`、`uint64_t` |

数据类型在头文件中以 `extern template` 声明、在实现文件中显式实例化。新增类型必须在 `Tensor.cpp`、`CpuTensor.cpp`、`CudaTensor.cu` 三处的实例化列表与三处头文件的声明列表中同步补齐，否则链接期报未定义符号。

---

## 核心语义

### 存储与视图

张量的实际位置由 `offset + Σ(index[i] * strides[i])` 确定。视图操作只改写 `shape` / `strides` / `offset`，通过 `shared_ptr` 与原张量共享同一块存储：

| 操作 | 是否复制 | 说明 |
| ---- | ---- | ---- |
| `Narrow` | 否 | 保留原 `strides`，仅移动 `offset` |
| `Transpose` / `Permute` | 否 | 就地交换 `strides`，返回 `bool` |
| `Squeeze` / `Unsqueeze` | 否 | 仅增删长度为 1 的维度 |
| `Expand` | 否 | 用 `stride = 0` 表达广播维度 |
| `View` | 否 | 仅接受连续张量，否则返回 `nullptr` |
| `Repeat` | 是 | 显式物化，重新分配连续存储 |
| `Clone` | 是 | 连续则直接拷贝，否则先连续化 |
| `Reshape` | 视情况 | 内部先 `Contiguous()`，因此总能成功 |
| `Contiguous` / `Contiguous_` | 视情况 | 已连续时返回自身 |

> **注意**：`View` 与 `Reshape` 的容错策略不同。`View` 对非连续张量返回 `nullptr`；`Reshape` 先做连续化，代价是一次拷贝但不会失败。

### 算术运算

- 逐元素加、减、乘、除要求 **设备相同且形状完全一致**，本库**不实现广播**；形状不符会抛出 `std::invalid_argument`。

- 需要广播时先显式 `Expand`，得到 `stride = 0` 的视图后再参与运算。

- 输出型接口返回新张量（`Add`、`Sub`、`Mul`、`Div`）；就地版本以 `_` 结尾（`Add_`、`Sub_`、`Mul_`、`Div_`）。

- 除法的除数含零时抛出 `std::runtime_error`，逐元素除与标量除均会检查。

- 标量重载以值传参（`Add(T value)`），与张量重载通过参数类型区分。

### 错误反馈方式

| 场景 | 行为 |
| ---- | ---- |
| `Expand` / `View` 参数非法 | 返回 `nullptr`，并向 stdout 打印原因 |
| 设备或形状不匹配 | 抛出 `std::invalid_argument` |
| 除数为零 | 抛出 `std::runtime_error` |
| 索引越界或维数不符 | 抛出 `std::out_of_range` / `std::invalid_argument` |

> **最佳实践**：调用 `Expand`、`View` 后必须判空；其余接口以异常报告错误，不要依赖返回值判错。

---

## API 速览

### 构造与工厂

| 接口 | 说明 |
| ---- | ---- |
| `CpuTensor<T>::CreateCpuTensor(shape)` | 分配但未初始化 |
| `CudaTensor<T>::CreateCudaTensor(shape)` | 分配显存 |
| `Arrange(shape, start, step)` | 等差数列填充 |
| `Full(shape, value)` | 指定值填充 |
| `Ones(shape)` / `Zeros(shape)` | 全 1 / 全 0 |
| `CreateFromCpuTensor(tensor)` | 由 CPU 张量构造 CUDA 张量 |

`CpuTensor` 与 `CudaTensor` 的构造函数均接受裸指针 + `deleter`、或 `shared_ptr`，配合 `offset` 实现对外部存储的零拷贝包装。

### 布局与形状

形状与元信息查询：

`GetShape` / `GetStrides` / `GetOffset` / `GetDim` / `GetTotalElements` / `GetContiguousStrides`

连续性判定：

`IsContiguous` / `HasValidDenseLayout`

视图与形状变换：

`View` / `Reshape` / `Permute` / `Transpose` / `Squeeze` / `Unsqueeze` / `Expand` / `Repeat` / `Narrow` / `Contiguous` / `Clone`

### 元素访问

| 接口 | 说明 |
| ---- | ---- |
| `At(indices)` / `SetAt(indices, value)` | 带边界检查，支持任意维 |
| `operator[](int64_t)` | 仅限一维张量 |
| `operator()(i, j, k, ...)` | 变参下标，返回 `T&` |
| `CheckEqual(vector<T>)` | 逐元素比对，测试用 |
| `HasZero()` | 是否存在零元素 |
| `Print(pretty)` / `PrintInfo()` | 打印元素 / 打印元信息 |

### 设备搬运

| 接口 | 说明 |
| ---- | ---- |
| `ToDevice(DeviceType)` | CPU ↔ CUDA 双向搬运 |
| `CudaTensor<T>::ToHost()` | 显存 → 内存 |
| `CpuTensorConvert<Src, Dest>` | CPU 张量跨数据类型转换 |
| `CudaTensorConvert<Src, Dest>` | CUDA 张量跨数据类型转换 |

---

## 示例

### CPU 后端

```cpp
#include "CpuTensor.h"

#include <iostream>
#include <memory>

int main()
{
    using TensorInt = CpuTensor<int>;

    auto base = TensorInt::Arrange({2, 3}, 1);
    base->Print();

    auto transposed = base->Clone();
    transposed->Transpose(0, 1);
    transposed->Contiguous()->Print();

    auto window = base->Narrow(0, 1, 1);
    window->Print();

    window->Repeat(0, 3)->Print();

    base->Add(*TensorInt::Ones({2, 3}))->Print();

    return 0;
}
```

输出：

```text
[[1, 2, 3],
 [4, 5, 6]]
[[1, 4],
 [2, 5],
 [3, 6]]
[[4, 5, 6]]
[[4, 5, 6],
 [4, 5, 6],
 [4, 5, 6]]
[[2, 3, 4],
 [5, 6, 7]]
```

### CUDA 后端

设备相关接口位于 CUDA 头文件中，使用方需以 `.cu` 编译：

```cpp
#include "CpuTensor.h"
#include "CudaTensor.h"
#include "TensorConvert.cuh"

#include <memory>

int main()
{
    auto host = CpuTensor<int>::Arrange({2, 3}, 1);

    auto device = host->ToDevice(DeviceType::CUDA);
    device->PrintInfo();

    auto back = std::dynamic_pointer_cast<CudaTensor<int>>(device)->ToHost();
    back->Print();

    auto converted = CudaTensorConvert<int, uint32_t>(
        std::dynamic_pointer_cast<CudaTensor<int>>(device).get());
    converted->Print();

    return 0;
}
```

输入为 `[[1, 2, 3], [4, 5, 6]]`（`int`）时，`converted` 为同形状、同值的 `uint32_t` 张量 `[[1, 2, 3], [4, 5, 6]]`。

---

## 已知限制

| 限制 | 说明 |
| ---- | ---- |
| 不支持广播 | 逐元素运算要求形状完全一致 |
| 无线性代数算子 | 未提供矩阵乘、归约、`sum` / `mean` 等 |
| 数据类型固定 | 仅上述 10 种，新增需同步实例化 |
| 视图受连续性约束 | `View` 对非连续张量返回 `nullptr` |
| `Tensor.h` 含 `using namespace std;` | 会污染包含方的全局命名空间 |

> **警告**：`Tensor.h` 在全局作用域写了 `using namespace std;`，任何包含它的翻译单元都会被注入 `std` 命名空间。新代码建议改为显式 `std::` 限定，此项改动属于公共 API 表面变更，需先确认。

---

## 许可证

BSD 4-Clause License，详见 [LICENSE](LICENSE)。Copyright (c) 2026, xunmeng2002。
