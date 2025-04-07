cmake_minimum_required(VERSION 3.20)

project(raytracer)

# project(raytracer CXX CUDA)

# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(PROJECT_PATH="${CMAKE_SOURCE_DIR}")
add_compile_definitions(SHADER_PATH="${CMAKE_SOURCE_DIR}/shaders")

include_directories(src)

file(GLOB_RECURSE INC_LIST
    src/*.hpp
)

file(GLOB_RECURSE SRC_LIST
    src/*.cpp
)

add_executable(raytracer ${INC_LIST} ${SRC_LIST})

# find_package(CUDAToolkit REQUIRED)
# find_package(OptiXToolkit REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(raytracer
#     cuda
#     OptiX::OptiX
    Vulkan::Vulkan
    glfw
)
