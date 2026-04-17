SET(PROJECT_NAME "Tensor")
project(${PROJECT_NAME} LANGUAGES CUDA CXX)
message("PROJECT_NAME:" ${PROJECT_NAME})


add_package(Tensor ${CMAKE_SOURCE_DIR}/Source/Tensor)


add_library(${PROJECT_NAME} SHARED ${all_file})

set(CMAKE_CUDA_ARCHITECTURES "75;86")
find_package(CUDAToolkit 13.0.0 REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE cudart.lib)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4819>)
    target_compile_options(${PROJECT_NAME} PRIVATE  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -g -O0>)
endif()


if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set_target_properties(${PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_compile_options(Tensor PRIVATE -Wno-ignored-attributes)
endif()


