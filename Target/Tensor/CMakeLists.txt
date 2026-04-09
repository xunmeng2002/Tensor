SET(PROJECT_NAME "Tensor")
project(${PROJECT_NAME} CUDA CXX)
message("PROJECT_NAME:" ${PROJECT_NAME})

set(CUDA_INCLUDE_DIR $ENV{CUDA_PATH}/include)
set(CUDA_LIB_DIR $ENV{CUDA_PATH}/lib/x64)
set(CMAKE_CUDA_ARCHITECTURES "75;86")

add_package(Tensor ${CMAKE_SOURCE_DIR}/Source/Tensor)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

add_library(${PROJECT_NAME} SHARED ${all_file})

target_compile_options(${PROJECT_NAME} PRIVATE  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -g -O0>)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4819>)

set_target_properties(${PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)


