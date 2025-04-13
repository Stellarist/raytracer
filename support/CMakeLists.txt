find_package(OpenGL REQUIRED)

set(GLFW_BUILD_DOCS OFF)
set(GLFW_BUILD_WAYLAND OFF)

add_subdirectory( glfw-3.4 )
set_property( TARGET glfw PROPERTY C_STANDARD 99 )
add_subdirectory( imgui )
add_subdirectory( miniz )

add_library(glad 
SHARED
    KHR/khrplatform.h
    glad/glad.c
    glad/glad.h
)

target_compile_definitions(glad
PRIVATE 
	GLAD_GLAPI_EXPORT_BUILD
PUBLIC  
	GLAD_GLAPI_EXPORT
)

target_include_directories(glad 
PUBLIC 
	. 
)

target_link_libraries(glad
PUBLIC 
	${OPENGL_LIBRARIES} 
)

set_property( TARGET glad PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
set_property( TARGET glfw PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
set_property( TARGET imgui PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
set_property( TARGET miniz PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
