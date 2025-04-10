#pragma once

#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#define CUDA_CHECK(call)                                                               \
{                                                                                      \
	cudaError_t rc = call;                                                       \
    if(rc != cudaSuccess) {                                                            \
        std::stringstream ss;                                                          \
        ss << "CUDA call (" << #call << " ) failed with error: '"                      \
           << cudaGetErrorString(rc) << "' (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw std::runtime_error(ss.str());                                            \
    }                                                                                  \
}

#define CUDA_SYNC_CHECK()                                                              \
{                                                                                      \
    cudaDeviceSynchronize();                                                           \
    cudaError_t rc = cudaGetLastError();                                               \
    if(rc != cudaSuccess) {                                                            \
        std::stringstream ss;                                                          \
        ss << "CUDA error on synchronize with error '"                                 \
           << cudaGetErrorString(rc) << "' (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw std::runtime_error(ss.str());                                            \
    }                                                                                  \
}

#define OPTIX_CHECK(call)                                                            \
{                                                                                    \
    OptixResult rc = call;                                                           \
    if (rc != OPTIX_SUCCESS) {                                                       \
        std::stringstream ss;                                                        \
        ss << "OptiX call (" << #call << ") failed with error: '"                    \
        << optixGetErrorString(rc) << "' (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw std::runtime_error(ss.str());                                          \
    }                                                                                \
}
