#include "Texture.hpp"

#include "stb/stb_image.h"

Texture::Texture(std::string tex_name, unsigned char* data, int w, int h, int c)
	: width(w), height(h), components(c), name(tex_name)
{
	tex_data.resize(width * height * components);
	std::copy(data, data + width * height * components, tex_data.begin());
}

bool Texture::LoadTexture(const std::string& filename)
{
	name = filename;
	components = 4;
	unsigned char* data = stbi_load(filename.c_str(), &width, &height, NULL, components);

	if (data == nullptr)
		return false;

	tex_data.resize(width * height * components);
	std::copy(data, data + width * height * components, tex_data.begin());
	stbi_image_free(data);

	return true;
}
