# PROGRESS

本文件记录跨会话状态。**会话开始时先读本文件**；已关闭条目移入 `PROGRESS-archive.md`（本仓库尚未建立归档文件）。

## 项目定位

C++20 的 N 维张量库，语义对齐 PyTorch，CPU（`CpuTensor`）与 CUDA（`CudaTensor`）双后端，抽象基类 `Tensor<T>` 统一接口。构建产物为动态库 `Tensor` 与测试程序 `TestTensor`。

---

## ✅ 已完成

### 2026-09-20 补充中英文 README

- 新增 `README.md`（中文）与 `README.en.md`（英文，含 `[中文文档]` 互链），覆盖特性、环境要求、构建、运行测试、目录结构、支持数据类型、核心语义、API 速览、示例、持续集成、已知限制、许可证。

- 示例代码与构建步骤均经实际验证：本机 MSVC 2022 Enterprise + CUDA 13.2 下 `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` 配置成功、`cmake --build --config Release` 构建成功、`TestTensor.exe` 运行退出码 0；README 中 CPU 示例的输出与实测逐行一致。

- 记录了两处需与代码侧对齐的事项：`Tensor.h` 的全局 `using namespace std;`，以及 CI 配置与 CMake 要求不符（见 ❓ 待讨论）。

---

## 🔄 进行中

无。

---

## ❓ 待讨论

- **Q.01 `Tensor.h` 的 `using namespace std;`（Tensor.h:12）**：违反 `rules/cpp-style.md` §5「禁止使用 using namespace std，尤其在头文件中」，且会污染所有包含方的全局命名空间。已在 README 的已知限制中如实标注。移除属公共 API 表面变更，按 Harness §3 须先经确认。

- **Q.02 CI 配置与构建要求不符**：`.workflow/pipeline.yml` 使用 `nvidia/cuda:12.0.0-devel-ubuntu22.04` 镜像，而 `Target/Tensor/CMakeLists.txt` 要求 `find_package(CUDAToolkit 13.0.0 REQUIRED)`；该流水线也只构建、不运行测试。`.gitee-ci.yml` 则通过 vcpkg 安装 `opencv` 与 `boost-system`，二者均非本项目依赖。两份配置如何取舍待定。

- **Q.03 根 CMakeLists.txt 的外部路径引用**：第 59–64 行引用 `${CMAKE_SOURCE_DIR}/../Libs/PersonalLib` 的 include 与 link 目录，该目录在当前工作区不存在（`D:/Gitee/Libs/PersonalLib` 缺失），实测不影响构建。保留还是清理待定。

---

## 备注

- 数据类型固定为 10 种（`float`、`double`、`int8_t`/`uint8_t`、`int16_t`/`uint16_t`、`int32_t`/`uint32_t`、`int64_t`/`uint64_t`）。新增类型须同步三处头文件的 `extern template` 声明与三处实现文件的实例化列表。

- 逐元素算子在代码层（`CheckDevice` / `CheckShape`）强制要求设备与形状完全一致，不实现广播，形状不符抛 `std::invalid_argument`；`Expand` 与 `View` 则以返回 `nullptr` 报告失败。这是本库与 PyTorch 的主要语义差异。

- 本地验证环境：MSVC 2022 Enterprise（D 盘）、CUDA Toolkit 13.2、CMake 随 VS 附带。
