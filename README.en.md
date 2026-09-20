# Tensor

[中文文档](README.md)

A C++20 N-dimensional tensor library with PyTorch-aligned semantics: layout is described by `shape` + `strides` + `offset`, and view operations share the underlying storage. It ships two backends, CPU (`CpuTensor`) and CUDA (`CudaTensor`), unified behind the abstract base class `Tensor<T>`, which dispatches by device.

---

## Features

- **One interface, two backends**: the CPU and CUDA backends derive from the same abstract base class with identical operator signatures; `ToDevice` / `ToHost` move data across devices.

- **PyTorch-style layout semantics**: `Permute`, `Transpose`, `Narrow`, `Squeeze`, `Unsqueeze`, `Expand`, and `View` return views over shared storage without copying; writing through a view is visible in the original tensor.

- **Explicit template instantiation**: ten data types are supported, with implementations compiled in one place to keep template bloat and build times under control.

- **No third-party dependencies**: nothing beyond the CUDA Runtime.

- **Executable reference tests**: the test project builds alongside the library, and its expected values are aligned with the PyTorch reference script `TestTensor.py` for case-by-case comparison.

---

## Requirements

| Item | Requirement |
| ---- | ---- |
| CMake | 3.20 or newer |
| C++ standard | C++20 |
| CUDA Toolkit | 13.0.0 or newer |
| CUDA architectures | 75, 86 (Turing / Ampere) |
| Windows compiler | MSVC with `/W4 /bigobj` |
| Linux compiler | GCC 13.4 (matches CI) |
| Third-party dependencies | none (CUDA Runtime only) |

> **Note**: The CUDA Toolkit is a hard dependency. `find_package(CUDAToolkit 13.0.0 REQUIRED)` in `Target/Tensor/CMakeLists.txt` fails configuration outright, so the toolkit must be installed even when only the CPU backend is used.

> **Note**: The root `CMakeLists.txt` references include and link directories under `${CMAKE_SOURCE_DIR}/../Libs/PersonalLib`, but no target links against them. A missing directory does not break the build.

---

## Building

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On Windows the Visual Studio generator is recommended (`-A x64` selects the 64-bit platform explicitly; on an x64 host that is already the default, so it can be omitted):

```shell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
```

The bundled `CMakeSettings.json` presets can also be used directly (Visual Studio "Open a CMake project"):

| Preset | Generator | Build type | Environment |
| ---- | ---- | ---- | ---- |
| `x64-Debug` | Ninja | Debug | MSVC x64 |
| `x64-Release` | Ninja | RelWithDebInfo | MSVC x64 |
| `WSL-GCC-Debug` | Ninja | Debug | WSL Linux x64 |

Build outputs:

| Platform | Library | Test executable |
| ---- | ---- | ---- |
| Linux | `build/Target/Tensor/libTensor.so` | `build/Test/TestTensor/TestTensor` |
| Windows | `build/Target/Tensor/Tensor.dll` | `build/Test/TestTensor/TestTensor.exe` |

With a multi-config generator (such as `Visual Studio 17 2022`), a configuration subdirectory is inserted into those paths, for example `build/Test/TestTensor/Release/TestTensor.exe`.

On Windows a post-build step copies `Tensor.dll` next to the executable, so no `PATH` configuration is needed.

`Tensor` enables `WINDOWS_EXPORT_ALL_SYMBOLS` on Windows, so no `__declspec(dllexport)` annotations are required.

---

## Running the tests

```shell
./build/Test/TestTensor/TestTensor
```

`Test/TestTensor/Src/Main.cpp` switches the device under test with a single variable:

```cpp
DeviceType device = DeviceType::CPU;
//DeviceType device = DeviceType::CUDA;
```

Coverage includes layout operations (transpose / permute / view / squeeze / expand / repeat / narrow), contiguity, cloning, element-wise arithmetic, cross-dtype conversion, and device transfers. Every case asserts `offset`, `shape`, `strides`, `IsContiguous`, and all element values.

---

## Repository layout

```text
Tensor/
├── Source/Tensor/         # Library sources
├── Target/Tensor/         # CMake definition of the library target
├── Test/TestTensor/       # Test executable
├── TestTensor.py          # PyTorch reference script
├── ConvertToUtf8Bom.py    # UTF-8 BOM converter (keeps line endings)
├── CMakeSettings.json     # Visual Studio CMake presets
└── .workflow/pipeline.yml # Gitee Go pipeline definition
```

Responsibilities of the core sources:

| File | Responsibility |
| ---- | ---- |
| `Tensor.h` / `Tensor.cpp` | Abstract base, layout math, views, dispatch |
| `CpuTensor.h` / `CpuTensor.cpp` | CPU backend and factory methods |
| `CudaTensor.h` / `CudaTensor.cu` | CUDA backend and device memory |
| `CudaTensorKernel.cuh` / `.cu` | Device kernels |
| `TensorConvert.cuh` | Cross-dtype conversion (`CpuTensorConvert` / `CudaTensorConvert`) |

---

## Supported data types

| Category | Types |
| ---- | ---- |
| Floating point | `float`, `double` |
| Signed integers | `int8_t`, `int16_t`, `int32_t`, `int64_t` |
| Unsigned integers | `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t` |

Types are declared with `extern template` in the headers and instantiated explicitly in the implementation files. Adding a type requires updating all three instantiation lists (`Tensor.cpp`, `CpuTensor.cpp`, `CudaTensor.cu`) and all three declaration lists in the headers, otherwise the link fails with undefined symbols.

---

## Core semantics

### Storage and views

An element's location is `offset + Σ(index[i] * strides[i])`. View operations only rewrite `shape` / `strides` / `offset` and share storage with the original tensor through a `shared_ptr`:

| Operation | Copies? | Notes |
| ---- | ---- | ---- |
| `Narrow` | No | Keeps the original `strides`, moves `offset` |
| `Transpose` / `Permute` | No | Swaps `strides` in place, returns `bool` |
| `Squeeze` / `Unsqueeze` | No | Adds or removes size-1 dimensions |
| `Expand` | No | Expresses broadcast dims with `stride = 0` |
| `View` | No | Contiguous tensors only, else `nullptr` |
| `Repeat` | Yes | Materializes into fresh contiguous storage |
| `Clone` | Yes | Copies when contiguous, else makes contiguous first |
| `Reshape` | Sometimes | Calls `Contiguous()` first, so it always succeeds |
| `Contiguous` / `Contiguous_` | Sometimes | Returns itself when already contiguous |

> **Note**: `View` and `Reshape` make different trade-offs. `View` returns `nullptr` for a non-contiguous tensor; `Reshape` makes a contiguous copy first, so it costs a copy but never fails.

### Arithmetic

- Element-wise add, subtract, multiply, and divide require **the same device and exactly the same shape**. This library **does not broadcast**; a shape mismatch throws `std::invalid_argument`.

- To emulate broadcasting, call `Expand` explicitly, then operate on the resulting `stride = 0` view.

- Out-of-place operators return a new tensor (`Add`, `Sub`, `Mul`, `Div`); in-place variants end with `_` (`Add_`, `Sub_`, `Mul_`, `Div_`).

- Division throws `std::runtime_error` when the divisor contains a zero; this check covers both tensor and scalar divisors.

- Scalar overloads take the value by value (`Add(T value)`) and are distinguished from tensor overloads by parameter type.

### How errors are reported

| Situation | Behavior |
| ---- | ---- |
| Invalid `Expand` / `View` arguments | Returns `nullptr`, prints the reason to stdout |
| Device or shape mismatch | Throws `std::invalid_argument` |
| Division by zero | Throws `std::runtime_error` |
| Out-of-range or wrong-rank index | Throws `std::out_of_range` / `std::invalid_argument` |

> **Best practice**: Always null-check the result of `Expand` and `View`. All other interfaces report errors by exception, so do not test their return values for failure.

---

## API overview

### Construction and factories

| Interface | Description |
| ---- | ---- |
| `CpuTensor<T>::CreateCpuTensor(shape)` | Allocates, uninitialized |
| `CudaTensor<T>::CreateCudaTensor(shape)` | Allocates device memory |
| `Arrange(shape, start, step)` | Fills with an arithmetic sequence |
| `Full(shape, value)` | Fills with a value |
| `Ones(shape)` / `Zeros(shape)` | All ones / all zeros |
| `CreateFromCpuTensor(tensor)` | Builds a CUDA tensor from a CPU one |

Both `CpuTensor` and `CudaTensor` constructors accept either a raw pointer plus a `deleter`, or a `shared_ptr`, together with an `offset`, which allows zero-copy wrapping of external storage.

### Layout and shape

Shape and metadata queries:

`GetShape` / `GetStrides` / `GetOffset` / `GetDim` / `GetTotalElements` / `GetContiguousStrides`

Contiguity checks:

`IsContiguous` / `HasValidDenseLayout`

Views and shape transforms:

`View` / `Reshape` / `Permute` / `Transpose` / `Squeeze` / `Unsqueeze` / `Expand` / `Repeat` / `Narrow` / `Contiguous` / `Clone`

### Element access

| Interface | Description |
| ---- | ---- |
| `At(indices)` / `SetAt(indices, value)` | Bounds-checked, any rank |
| `operator[](int64_t)` | 1-D tensors only |
| `operator()(i, j, k, ...)` | Variadic subscript, returns `T&` |
| `CheckEqual(vector<T>)` | Element-wise comparison, for tests |
| `HasZero()` | Whether any element is zero |
| `Print(pretty)` / `PrintInfo()` | Prints elements / prints metadata |

### Device transfer

| Interface | Description |
| ---- | ---- |
| `ToDevice(DeviceType)` | Moves both ways between CPU and CUDA |
| `CudaTensor<T>::ToHost()` | Device memory to host memory |
| `CpuTensorConvert<Src, Dest>` | Cross-dtype conversion on CPU |
| `CudaTensorConvert<Src, Dest>` | Cross-dtype conversion on CUDA |

---

## Examples

### CPU backend

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

Output:

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

### CUDA backend

Device-specific interfaces live in the CUDA headers, so this translation unit must be compiled as `.cu`:

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

Given the input `[[1, 2, 3], [4, 5, 6]]` (`int`), `converted` is a `uint32_t` tensor of the same shape and values: `[[1, 2, 3], [4, 5, 6]]`.

---

## Continuous integration

`.workflow/pipeline.yml` defines a Gitee Go pipeline that runs inside the `nvidia/cuda:12.0.0-devel-ubuntu22.04` image with GCC 13.4:

```shell
mkdir build && cd build
cmake -G "Unix Makefiles" ../ && make -j
```

The pipeline **only builds; it does not run the tests** (no `./Test/TestTensor` invocation).

> **Note**: The pipeline image ships CUDA 12.0, which is older than the `find_package(CUDAToolkit 13.0.0 REQUIRED)` requirement in `Target/Tensor/CMakeLists.txt`. Whether this pipeline can actually run needs verification.

The repository also contains `.gitee-ci.yml`, whose script installs `opencv` and `boost-system` through vcpkg before building and running `./Test/TestTensor`. Neither library is a dependency of this project, so that file looks like a leftover or copied template.

---

## Known limitations

| Limitation | Description |
| ---- | ---- |
| No broadcasting | Element-wise ops require identical shapes |
| No linear algebra | No matmul, reductions, `sum` / `mean` |
| Fixed set of dtypes | Only the ten listed above |
| Views need contiguity | `View` returns `nullptr` otherwise |
| `Tensor.h` has `using namespace std;` | Pollutes includers' global namespace |

> **Warning**: `Tensor.h` declares `using namespace std;` at global scope, so every translation unit that includes it has the `std` namespace injected. New code should use explicit `std::` qualification; changing this alters the public API surface and should be confirmed first.

---

## License

BSD 4-Clause License, see [LICENSE](LICENSE). Copyright (c) 2026, xunmeng2002.
