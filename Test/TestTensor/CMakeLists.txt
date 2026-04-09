SET(PROJECT_NAME "TestTensor")
project(${PROJECT_NAME} CUDA CXX)
message("PROJECT_NAME:" ${PROJECT_NAME})

set(CUDA_INCLUDE_DIR $ENV{CUDA_PATH}/include)
set(CUDA_LIB_DIR $ENV{CUDA_PATH}/lib/x64)
set(CMAKE_CUDA_ARCHITECTURES "75;86")

add_package(Src ./Src)

include_directories(${CUDA_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Source/Tensor)
link_directories(${CUDA_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/out/build/x64-Debug/Target/Tensor)

add_executable(${PROJECT_NAME} ${all_file})

target_link_libraries(${PROJECT_NAME} Tensor.lib)

# 为 Debug 模式添加 CUDA 调试符号
target_compile_options(${PROJECT_NAME} PRIVATE  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -g -O0>)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4819>)

add_custom_command(
TARGET ${PROJECT_NAME}
POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy
	"$<TARGET_FILE:Tensor>" 
	"$<TARGET_FILE_DIR:${PROJECT_NAME}>/$<TARGET_FILE_NAME:Tensor>"
COMMENT "Copying Tensor.dll DLL...")


