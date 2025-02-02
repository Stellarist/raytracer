cmake_minimum_required(VERSION 3.20)

project(raytracer)

add_compile_definitions(PROJECT_PATH="${CMAKE_SOURCE_DIR}")

include_directories(src)
include_directories(thirdparty)

file(GLOB SRC_LIST src/*.cpp)
file(GLOB SHADER_LIST shaders/*)

add_executable(raytracer ${SRC_LIST})

find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(raytracer
    glad::glad
    glm::glm
)
