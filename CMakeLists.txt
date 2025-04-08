cmake_minimum_required(VERSION 3.20)

project(OptiXDemo CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

file(GLOB_RECURSE INC_LIST src/*.hpp src/*.h)
file(GLOB_RECURSE SRC_LIST src/*.cpp src/*.cu)

find_package(OptiX REQUIRED)

add_executable(OptiXDemo
    ${INC_LIST}
    ${SRC_LIST}
)

target_include_directories(OptiXDemo 
PRIVATE
    thirdparty
    ${OptiX_INCLUDE}
)

target_link_libraries(OptiXDemo
PRIVATE
    ${CUDA_LIBRARIES}
    ${OptiX_LIBRARY}
    glad::glad
    glfw3
)
