#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <stb_image_write.h>

#include "debug.hpp"
#include "SampleRenderer.hpp"
#include "utils/CUDABuffer.hpp"

int main(int argc, char const *argv[])
{
    try{
		SampleRenderer sample;

		// const glm::ivec2 fbSize(glm::ivec2(1200, 1024));
		const ivec2 fbSize{1200, 1024};
		sample.resize(fbSize);
		sample.render();

		std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
		sample.downloadPixels(pixels.data());

		const std::string fileName = "osc_example2.png";
		stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
		               pixels.data(), fbSize.x * sizeof(uint32_t));
		std::cout  << std::endl
		          << "Image rendered, and saved to " << fileName << " ... done." << std::endl
		          << std::endl;
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}

    return 0;
}

