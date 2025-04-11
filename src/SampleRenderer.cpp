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

#include "SampleRenderer.hpp"

#include <iostream>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "debug.hpp"
#include "utils/PTXCode.hpp"

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
SampleRenderer::SampleRenderer()
{
  initOptix();
    
  std::cout << "#osc: creating optix context ..." << std::endl;
  createContext();
    
  std::cout << "#osc: setting up module ..." << std::endl;
  createModule();

  std::cout << "#osc: creating raygen programs ..." << std::endl;
  createRaygenPrograms();
  std::cout << "#osc: creating miss programs ..." << std::endl;
  createMissPrograms();
  std::cout << "#osc: creating hitgroup programs ..." << std::endl;
  createHitgroupPrograms();

  std::cout << "#osc: setting up optix pipeline ..." << std::endl;
  createPipeline();

  std::cout << "#osc: building SBT ..." << std::endl;
  buildSBT();

  launchParamsBuffer.alloc(1);
  std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

  std::cout << "#osc: Optix Sample fully set up" << std::endl;
}

/*! helper function that initializes optix and checks for errors */
void SampleRenderer::initOptix()
{
  std::cout << "#osc: initializing optix..." << std::endl;
    
  // check for available optix7 capable devices
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("#osc: no CUDA capable devices found!");
  std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

  // initialize optix
  OPTIX_CHECK( optixInit() );
           std::cout << "#osc: successfully initialized optix... yay!"
            << std::endl;
}

static void context_log_cb(unsigned int level,
                            const char *tag,
                            const char *message,
                            void *)
{
  fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}

/*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
void SampleRenderer::createContext()
{
  // for this sample, do everything on one device
  const int deviceID = 0;
  CUDA_CHECK(cudaSetDevice(deviceID));
  CUDA_CHECK(cudaStreamCreate(&stream));
    
  cudaGetDeviceProperties(&deviceProps, deviceID);
  std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
    
  CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
  if( cuRes != CUDA_SUCCESS ) 
    fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
    
  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback
              (optixContext,context_log_cb,nullptr,4));
}

/*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
void SampleRenderer::createModule()
{
  moduleCompileOptions.maxRegisterCount  = 50;
  moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipelineCompileOptions = {};
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur     = false;
  pipelineCompileOptions.numPayloadValues   = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "launch_params";
    
  pipelineLinkOptions.maxTraceDepth          = 2;
    
  std::string ptxCode;
  PTXCode::readFromFile("cuda_compile_ptx_1_generated_main.cu.ptx", ptxCode);
    
  char log[2048];
  size_t sizeof_log = sizeof( log );
#if OPTIX_VERSION >= 70700
  OPTIX_CHECK(optixModuleCreate(optixContext,
                                        &moduleCompileOptions,
                                        &pipelineCompileOptions,
                                        ptxCode.c_str(),
                                        ptxCode.size(),
                                        log,&sizeof_log,
                                        &module
                                        ));
#else
  OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                        &moduleCompileOptions,
                                        &pipelineCompileOptions,
                                        ptxCode.c_str(),
                                        ptxCode.size(),
                                        log,      // Log string
                                        &sizeof_log,// Log string sizse
                                        &module
                                        ));
#endif
  if (sizeof_log > 1)
	  std::cout << log << std::endl;
}
  
/*! does all setup for the raygen program(s) we are going to use */
void SampleRenderer::createRaygenPrograms()
{
  // we do a single ray gen program in this example:
  raygenPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module            = module;           
  pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

  // OptixProgramGroup raypg;
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &raygenPGs[0]
                                      ));
  if (sizeof_log > 1) 
	  std::cout << log << std::endl;
}
  
/*! does all setup for the miss program(s) we are going to use */
void SampleRenderer::createMissPrograms()
{
  // we do a single ray gen program in this example:
  missPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module            = module;           
  pgDesc.miss.entryFunctionName = "__miss__radiance";

  // OptixProgramGroup raypg;
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &missPGs[0]
                                      ));
  if (sizeof_log > 1) 
	  std::cout << log << std::endl;
}
  
/*! does all setup for the hitgroup program(s) we are going to use */
void SampleRenderer::createHitgroupPrograms()
{
  // for this simple example, we set up a single hit group
  hitgroupPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgDesc.hitgroup.moduleCH            = module;           
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgDesc.hitgroup.moduleAH            = module;           
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &hitgroupPGs[0]
                                      ));
  if (sizeof_log > 1) 
	  std::cout << log << std::endl;
}
  

/*! assembles the full pipeline of all programs */
void SampleRenderer::createPipeline()
{
  std::vector<OptixProgramGroup> programGroups;
  for (auto pg : raygenPGs)
    programGroups.push_back(pg);
  for (auto pg : missPGs)
    programGroups.push_back(pg);
  for (auto pg : hitgroupPGs)
    programGroups.push_back(pg);
    
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixPipelineCreate(optixContext,
                                  &pipelineCompileOptions,
                                  &pipelineLinkOptions,
                                  programGroups.data(),
                                  (int)programGroups.size(),
                                  log,&sizeof_log,
                                  &pipeline
                                  ));
  if (sizeof_log > 1) 
	  std::cout << log << std::endl;

  OPTIX_CHECK(optixPipelineSetStackSize
              (/* [in] The pipeline to configure the stack size for */
                pipeline, 
                /* [in] The direct stack size requirement for direct
                  callables invoked from IS or AH. */
                2*1024,
                /* [in] The direct stack size requirement for direct
                  callables invoked from RG, MS, or CH.  */                 
                2*1024,
                /* [in] The continuation stack requirement. */
                2*1024,
                /* [in] The maximum depth of a traversable graph
                  passed to trace. */
                1));
  if (sizeof_log > 1) 
	  std::cout << log << std::endl;
}

/*! constructs the shader binding table */
void SampleRenderer::buildSBT()
{
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for (int i=0;i<raygenPGs.size();i++) {
    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);
  }
  raygenRecordsBuffer.alloc(raygenRecords.size());
  raygenRecordsBuffer.upload(raygenRecords.data());
  sbt.raygenRecord = raygenRecordsBuffer.getDevicePtr();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for (int i=0;i<missPGs.size();i++) {
    MissRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
    rec.data = nullptr; /* for now ... */
    missRecords.push_back(rec);
  }
  missRecordsBuffer.alloc(missRecords.size());
  missRecordsBuffer.upload(missRecords.data());
  sbt.missRecordBase          = missRecordsBuffer.getDevicePtr();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount         = (int)missRecords.size();

  // we don't actually have any objects in this example, but let's
  // create a dummy one so the SBT doesn't have any null pointers
  // (which the sanity checks in compilation would complain about)
  int numObjects = 1;
  std::vector<HitgroupRecord> hitgroupRecords;
  for (int i=0;i<numObjects;i++) {
    int objectType = 0;
    HitgroupRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType],&rec));
    rec.objectID = i;
    hitgroupRecords.push_back(rec);
  }
  hitgroupRecordsBuffer.alloc(hitgroupRecords.size());
  hitgroupRecordsBuffer.upload(hitgroupRecords.data());
  sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.getDevicePtr();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
}

/*! render one frame */
void SampleRenderer::render()
{
  // sanity check: make sure we launch only after first resize is
  // already done:
  if (launchParams.fb_size.x == 0) return;

  launchParamsBuffer.upload(&launchParams);
  launchParams.frame_id++;
    
  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipeline,stream,
                          /*! parameters and SBT */
                          launchParamsBuffer.getDevicePtr(),
                          launchParamsBuffer.byteSize(),
                          &sbt,
                          /*! dimensions of the launch: */
                          launchParams.fb_size.x,
                          launchParams.fb_size.y,
                          1
                          ));
  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  CUDA_SYNC_CHECK();
}

/*! resize frame buffer to given resolution */
// void SampleRenderer::resize(const glm::ivec2 &newSize)
void SampleRenderer::resize(const ivec2 &newSize)
{
  // if window minimized
  if (newSize.x == 0 || newSize.y == 0) return;
  
  // resize our cuda frame buffer
  colorBuffer.resize(newSize.x*newSize.y);

  // update the launch parameters that we'll pass to the optix
  // launch:
  launchParams.fb_size      = newSize;
  launchParams.color_buffer = (uint32_t*)colorBuffer.get();
}

/*! download the rendered color buffer */
void SampleRenderer::downloadPixels(uint32_t h_pixels[])
{
  colorBuffer.download(h_pixels);
}
