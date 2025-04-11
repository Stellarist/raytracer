cmake_minimum_required(VERSION 3.20)

project(raytracer CXX CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/configure_optix.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE INC_LIST src/*.hpp src/*.h)
file(GLOB_RECURSE SRC_LIST src/*.cpp)

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)
find_package(glad REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Stb REQUIRED)

cuda_compile_and_embed(embedded_ptx_code src/main.cu)

add_executable(raytracer
    ${embedded_ptx_code}
    ${SRC_LIST}
    ${INC_LIST}
)

target_include_directories(raytracer 
PRIVATE
    src
    thirdparty
    ${OptiX_INCLUDE}
)

target_link_libraries(raytracer
PRIVATE
    ${OptiX_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    glm::glm-header-only
    # glad::glad
    # glfw3
)

set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD_REQUIRED ON
)
