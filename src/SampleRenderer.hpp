// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <vector>
// #include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "utils/CUDABuffer.hpp"
#include "LaunchParams.hpp"

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  int objectID;
};

class SampleRenderer {
public:
  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer();

  /*! render one frame */
  void render();

  /*! resize frame buffer to given resolution */
  // void resize(const glm::ivec2 &newSize);
  void resize(const ivec2 &newSize);

  /*! download the rendered color buffer */
  void downloadPixels(uint32_t h_pixels[]);

protected:
  /*! helper function that initializes optix and checks for errors */
  void initOptix();

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void createContext();

  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void createModule();
  
  /*! does all setup for the raygen program(s) we are going to use */
  void createRaygenPrograms();
  
  /*! does all setup for the miss program(s) we are going to use */
  void createMissPrograms();
  
  /*! does all setup for the hitgroup program(s) we are going to use */
  void createHitgroupPrograms();

  /*! assembles the full pipeline of all programs */
  void createPipeline();

  /*! constructs the shader binding table */
  void buildSBT();

protected:
  /*! @{ CUDA device context and stream that optix pipeline will run
      on, as well as device properties for this device */
  CUcontext          cudaContext;
  CUstream           stream;
  cudaDeviceProp     deviceProps;
  /*! @} */

  //! the optix context that our pipeline will run in.
  OptixDeviceContext optixContext;

  /*! @{ the pipeline we're building */
  OptixPipeline               pipeline;
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  OptixPipelineLinkOptions    pipelineLinkOptions    = {};
  /*! @} */

  /*! @{ the module that contains out device programs */
  OptixModule                 module;
  OptixModuleCompileOptions   moduleCompileOptions = {};
  /* @} */

  /*! vector of all our program(group)s, and the SBT built around
      them */
  OptixShaderBindingTable sbt = {};
  LaunchParams launchParams;

  std::vector<OptixProgramGroup> raygenPGs;
  std::vector<OptixProgramGroup> missPGs;
  std::vector<OptixProgramGroup> hitgroupPGs;

  CUDABuffer<RaygenRecord>   raygenRecordsBuffer;
  CUDABuffer<MissRecord>     missRecordsBuffer;
  CUDABuffer<HitgroupRecord> hitgroupRecordsBuffer;
  CUDABuffer<LaunchParams>   launchParamsBuffer;
  CUDABuffer<uint32_t>       colorBuffer;
};
