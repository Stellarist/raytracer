OPTIX_add_sample_executable(raytracer target_name
    raytracer.cu
    raytracer.cpp
    raytracer.h
    OPTIONS -rdc true
)

target_include_directories(${target_name}
PRIVATE
	${CUDA_LIBRARIES}
)

find_package(glad REQUIRED)
target_link_libraries(${target_name}
	${CUDA_LIBRARIES}
	glad::glad
	glfw
	imgui::imgui
	miniz::miniz
)
