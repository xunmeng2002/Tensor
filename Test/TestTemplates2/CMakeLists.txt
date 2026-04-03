# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.20)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

SET(PROJECT_NAME "TestTemplates2")
project(${PROJECT_NAME} CUDA CXX)
message("PROJECT_NAME:" ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(BUILD_TEST_PROJECT 1)

set(CUDA_INCLUDE_DIR $ENV{CUDA_PATH}/include)
set(CUDA_LIB_DIR $ENV{CUDA_PATH}/lib/x64)
set(CMAKE_CUDA_ARCHITECTURES "75;86")



macro(add_package name dir)
	include_directories(${dir})
	set(SUFFIXES "*.h" "*.hpp" "*.hxx" "*.cpp" "*.c" "*.cu" "*.cuh")
	set(package_files "")
    foreach(suffix ${SUFFIXES})
        file(GLOB_RECURSE temp_files ${dir}/${suffix})
        list(APPEND package_files ${temp_files})
    endforeach()
    source_group(${ARGV0} FILES ${package_files})
    set(all_file ${all_file} ${package_files})
endmacro(add_package)

add_package(Source ${CMAKE_SOURCE_DIR}/Source)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

add_executable(${PROJECT_NAME} ${all_file})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4819>)




