cmake_minimum_required(VERSION 3.20)

project(raytracer CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cmake)
include(thirdparty/cmake/Macros.cmake)
include(thirdparty/cmake/ConfigureOptiX.cmake)

file(GLOB_RECURSE INC_LIST src/*.hpp src/*.h)
file(GLOB_RECURSE SRC_LIST src/*.cpp)

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)
find_package(glad REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Stb REQUIRED)

# add_executable(raytracer
#     ${embedded_ptx_code}
#     ${SRC_LIST}
#     ${INC_LIST}
# )

OPTIX_add_sample_executable(raytracer target_name
    ${SRC_LIST}
    ${INC_LIST}
    OPTIONS -rdc true
)

target_include_directories(${target_name}
PRIVATE
    src
    thirdparty
    ${OptiX_INCLUDE}
)

target_link_libraries(${target_name}
    ${OptiX_LIBRARY}
    ${CUDA_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    glm::glm
)
