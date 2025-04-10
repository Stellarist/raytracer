#include <cstdio>
#include <optix_device.h>

#include "LaunchParams.hpp"

__constant__ LaunchParams launch_params;

__global__ void __closetHitRadiance()
{
}

__global__ void __anyHitRadiance()
{
}

__global__ void __missRadiance()
{
}

__global__ void __raygenRenderFrame()
{
	if (launch_params.frame_id == 0 &&
        optixGetLaunchIndex().x == 0 &&
        optixGetLaunchIndex().y == 0) {

        printf("############################################\n");
        printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
		       launch_params.fb_size.x,
		       launch_params.fb_size.y);
        printf("############################################\n");
    }    
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int r = (ix % 256);
    const int g = (iy % 256);
    const int b = ((ix+iy) % 256);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
	const uint32_t fb_index = ix + iy * launch_params.fb_size.x;
	launch_params.color_buffer[fb_index] = rgba;
}
