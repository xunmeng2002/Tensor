SET(PROJECT_NAME "TestTensor")
project(${PROJECT_NAME})
message("PROJECT_NAME:" ${PROJECT_NAME})

add_package(Src ./Src)

include_directories(${CMAKE_SOURCE_DIR}/Source/Tensor)
link_directories(${CMAKE_SOURCE_DIR}/out/build/x64-Debug/Target/Tensor)

add_executable(${PROJECT_NAME} ${all_file})

target_link_libraries(${PROJECT_NAME} Tensor.lib)


add_custom_command(
TARGET ${PROJECT_NAME}
POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy
	"$<TARGET_FILE:Tensor>" 
	"$<TARGET_FILE_DIR:${PROJECT_NAME}>/$<TARGET_FILE_NAME:Tensor>"
COMMENT "Copying Tensor.dll DLL...")


