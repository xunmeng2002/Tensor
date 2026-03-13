SET(PROJECT_NAME "TestTensor")
project(${PROJECT_NAME})
message("PROJECT_NAME:" ${PROJECT_NAME})

add_package(Src ./Src)
add_package(Tensor ${CMAKE_SOURCE_DIR}/Source/Tensor)

add_executable(${PROJECT_NAME} ${all_file})



