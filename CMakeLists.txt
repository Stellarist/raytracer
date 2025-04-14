cmake_minimum_required(VERSION 3.20)

project(raytracer)

if( POLICY CMP0072 )
	cmake_policy(SET CMP0072 NEW)
endif()

if( POLICY CMP0074 )
	cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_MODULE_PATH
	"${CMAKE_SOURCE_DIR}/CMake"
	${CMAKE_MODULE_PATH}
)

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

option(OPTIX_DEBUG_DEVICE_CODE "When enabled, build OptiX-IR files for debugging." OFF)

if( OPTIX_DEBUG_DEVICE_CODE )
	set( OPTIX_OPTIXIR_BUILD_CONFIGURATION "Debug" )
else()
	set( OPTIX_OPTIXIR_BUILD_CONFIGURATION "Release" )
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

include(CMake/Macros.cmake)
include (CMake/CompilerInfo.cmake)
include (CMake/ConfigCompilerFlags.cmake)

option(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)
option(CUDA_USE_LOCAL_ENV "Pass the '--use-local-env' option to NVCC; only applies to initial configure" ON)

find_package(CUDA REQUIRED)

set( CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile OPTIXIR/PTX at run-time instead of NVCC at build-time" )
if( NOT ( NOT ( NOT CUDA_NVRTC_ENABLED ) ) EQUAL ( NOT ( NOT CUDA_NVRTC_ENABLED_INTERNAL ) ) )
	message( STATUS "Resetting OPTIXIR/PTX support" )
	unset( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT CACHE )
endif()
set( CUDA_NVRTC_ENABLED_INTERNAL ${CUDA_NVRTC_ENABLED} CACHE INTERNAL "Previous configured value (NVRTC)" FORCE )

if( CUDA_NVRTC_ENABLED )
	if( CUDA_VERSION VERSION_LESS 12.0 )
		if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
			message( SEND_ERROR "CUDA_NVRTC_ENABLED is not compatible with SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT in CUDA versions less than 12.0" )
		else()
			option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
		endif()
	else()
		option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
	endif()
else()
	if( CUDA_VERSION VERSION_LESS 11.7 )
		if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
			message( SEND_ERROR "SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7" )
		else()
			option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
		endif()
	else()
		option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
	endif()
endif()

if( NOT ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT ) ) EQUAL ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ) ) )
	message( STATUS "Unsetting values associated with OptiX code generation" )
	unset( SAMPLES_INPUT_GENERATE_OPTIXIR CACHE )
	unset( SAMPLES_INPUT_GENERATE_PTX CACHE )
	unset( GENERATE_DEBUG_DEVICE_CODE CACHE )
endif()
set(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ${SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT} CACHE INTERNAL "Previous configured value (OPTIXIR)" FORCE)

if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
	option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON )
	option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF )
else()
	option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF )
	option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )
endif()

option(RELEASE_USE_STATIC_CRT "Build using the static CRT library" ON)

function(replace_flag var old_flag new_flag)
	string(REPLACE "${old_flag}" "${new_flag}" ${var} ${${var}})
	set(${var} "${${var}}" CACHE STRING "Default compiler flags" FORCE)
endfunction()

function(replace_flags old_flag new_flag)
	foreach(build "" _DEBUG _MINSIZEREL _RELEASE _RELWITHDEBINFO)
		replace_flag(CMAKE_C_FLAGS${build}   "${old_flag}" "${new_flag}")
		replace_flag(CMAKE_CXX_FLAGS${build} "${old_flag}" "${new_flag}")
	endforeach()
endfunction()

if(WIN32)
	if(RELEASE_USE_STATIC_CRT)
		replace_flags("/MD" "/MT")
	else()
		replace_flags("/MT" "/MD")
	endif()
endif(WIN32)

mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

function(optix_add_cuda_flag_config config flag)
	string(TOUPPER "${config}" config)
	list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
	if(index EQUAL -1)
		list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
		set(CUDA_NVCC_FLAGS${config} ${CUDA_NVCC_FLAGS${config}} CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
	endif()
endfunction()

function(optix_add_cuda_flag flag)
	optix_add_cuda_flag_config( "" ${flag} )
endfunction()

if( NOT PASSED_FIRST_CONFIGURE )
	set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
	string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

	list(FIND CUDA_NVCC_FLAGS "-arch" index)
	if(index EQUAL -1)
		list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
	endif()

	optix_add_cuda_flag("--use_fast_math")
	optix_add_cuda_flag("-lineinfo")

	if( DEFINED CMAKE_CONFIGURATION_TYPES )
		set( OPTIX_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} )
	else()
		set( OPTIX_CONFIGURATION_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
	endif()

	foreach( config ${OPTIX_CONFIGURATION_TYPES} )
		if( ${config} STREQUAL "Debug" )
			optix_add_cuda_flag_config( _${config} "-G" )
			optix_add_cuda_flag_config( _${config} "-O0" )
		endif()
	endforeach()

	if( CUDA_VERSION VERSION_LESS "3.0" )
		optix_add_cuda_flag("--keep")
	endif()

	if(CUDA_VERSION VERSION_GREATER "11.0")
		optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
	endif()

	if(CUDA_USE_LOCAL_ENV)
		optix_add_cuda_flag("--use-local-env")
	endif()

	if(CMAKE_CXX_STANDARD EQUAL 11)
		set(SAMPLES_NVRTC_CXX "-std=c++11")
	else()
		set(SAMPLES_NVRTC_CXX "")
	endif()

	if( NOT DEFINED CMAKE_CONFIGURATION_TYPES )
		if( NOT CMAKE_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE_PREVIOUS )
			message( STATUS "Resetting CUDA_NVRTC_FLAGS" )
			unset( CUDA_NVRTC_FLAGS CACHE )
		endif()
		set( CMAKE_BUILD_TYPE_PREVIOUS ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Previous configured value (CMAKE_BUILD_TYPE)" FORCE )

		set( configs "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
		foreach( config ${configs} )
			if( ${config} STREQUAL "Debug" )
				set( SAMPLES_NVRTC_DEBUG "-G" )
			else()
				set( SAMPLES_NVRTC_DEBUG "-lineinfo" )
			endif()
			string( TOUPPER ${config} config_upper )
			set( CUDA_NVRTC_FLAGS_${config_upper} ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} ${SAMPLES_NVRTC_DEBUG} -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )

			if( ${config} STREQUAL ${CMAKE_BUILD_TYPE} )
				set( CUDA_NVRTC_FLAGS ${CUDA_NVRTC_FLAGS_${config_upper}} CACHE STRING "List of NVRTC options just for the samples" )
			endif()
		endforeach()
	else()
		set( CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -lineinfo -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
		set( CUDA_NVRTC_FLAGS_DEBUG ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -G -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
	endif()
endif()

mark_as_advanced(CUDA_NVRTC_FLAGS)

if(USING_WINDOWS_CL)
	list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX REQUIRED)

include_directories(${OptiX_INCLUDE})

if( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT AND SAMPLES_INPUT_GENERATE_OPTIXIR )
	message( SEND_ERROR "Must enable SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT to enable SAMPLES_INPUT_GENERATE_OPTIXIR" )
endif()

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

if( WIN32 )
	string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
endif( WIN32 )

set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
set(SAMPLES_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../SDK/support")
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \".\", ")

set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

set( config_suffixes "_RELEASE" "_DEBUG" )
foreach( config_suffix ${config_suffixes} )
	if( ${config_suffix} STREQUAL "_RELEASE" )
		set( config_suffix "" )
	endif()
	set(CUDA_NVRTC_OPTIONS${config_suffix})
	foreach(flag ${CUDA_NVRTC_FLAGS${config_suffix}})
		set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}} \\\n  \"${flag}\",")
	endforeach()
	set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}}")
endforeach()

configure_file(sampleConfig.h.in sampleConfig.h @ONLY)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"${CMAKE_BINARY_DIR}/include"
	${CMAKE_CURRENT_BINARY_DIR}
	${CUDA_INCLUDE_DIRS}
)

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

macro(OPTIX_sutil_compile_to_optix_input generated_files)
	if( NOT CUDA_NVRTC_ENABLED )
		if( SAMPLES_INPUT_GENERATE_OPTIXIR )
			message("sutil OPTIXIR")
			set(CUDA_BUILD_CONFIGURATION ${OPTIX_OPTIXIR_BUILD_CONFIGURATION})
			CUDA_WRAP_SRCS( sutil OPTIXIR generated_files2 ${ARGN} )
			unset(CUDA_BUILD_CONFIGURATION)
			list(APPEND ${generated_files} ${generated_files2})
		endif()
		if( SAMPLES_INPUT_GENERATE_PTX )
			message("sutil PTX")
			CUDA_WRAP_SRCS( sutil PTX generated_files3 ${ARGN} )
			list(APPEND ${generated_files} ${generated_files3})
		endif()
	endif()
	message("${generated_files} = ${${generated_files}}")
endmacro()

macro(OPTIX_add_source_groups)
	if( NOT CUDA_NVRTC_ENABLED )
		if( SAMPLES_INPUT_GENERATE_PTX )
			source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
		endif()
		if( SAMPLES_INPUT_GENERATE_OPTIXIR )
			source_group("OptixIR Files"  REGULAR_EXPRESSION ".+\\.optixir$")
		endif()
	endif()
	source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
endmacro()

function(OPTIX_add_sample_executable target_name_base target_name_var)
	set( target_name ${target_name_base} )
	set( ${target_name_var} ${target_name} PARENT_SCOPE )

	OPTIX_add_source_groups()
	CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

	set(cu_obj_source_files)
	set(cu_optix_source_files)
	foreach(file ${source_files})
		get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
		if(${_cuda_source_format} MATCHES "OBJ")
			list(APPEND cu_obj_source_files ${file})
		else()
			list(APPEND cu_optix_source_files ${file})
		endif()
	endforeach()

	CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )

	if( SAMPLES_INPUT_GENERATE_OPTIXIR )
		set(CUDA_BUILD_CONFIGURATION ${OPTIX_OPTIXIR_BUILD_CONFIGURATION})
		CUDA_WRAP_SRCS( ${target_name} OPTIXIR generated_files2 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
		unset(CUDA_BUILD_CONFIGURATION)
		list(APPEND generated_files ${generated_files2})
	endif()
	if( SAMPLES_INPUT_GENERATE_PTX AND NOT CUDA_NVRTC_ENABLED)
		CUDA_WRAP_SRCS( ${target_name} PTX generated_files3 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
		list(APPEND generated_files ${generated_files3})
	endif()

	add_executable(${target_name}
		${source_files}
		${generated_files}
		${cmake_options}
    )

	target_link_libraries(${target_name}
		sutil
	)

	set_target_properties(${target_name}
	PROPERTIES
		COMPILE_DEFINITIONS
		"OPTIX_SAMPLE_NAME_DEFINE=${target_name};OPTIX_SAMPLE_DIR_DEFINE=${target_name}"
	)
endfunction()

include_directories(C:/Data/repos/codes/raytracer/build/vcpkg_installed/x64-windows/include)
include_directories(thirdparty)
include_directories(sutil)

find_package(glad REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)
find_package(miniz REQUIRED)

add_subdirectory(src)
add_subdirectory(sutil)

set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")
