#pragma once

#include <vector>
#include <string>

class Texture
{
public:
	Texture() : width(0), height(0), components(0) {};
	Texture(std::string texName, unsigned char* data, int w, int h, int c);
	~Texture() {}

	bool LoadTexture(const std::string& filename);

	int width;
	int height;
	int components;
	std::string name;
	std::vector<unsigned char> tex_data;
};
