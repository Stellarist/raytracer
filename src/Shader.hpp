#pragma once

#include <string>

struct ShaderSource {
	std::string src;
	std::string path;
};

class Shader {
public:
	Shader(const ShaderSource& source_obj, unsigned int shader_type);
	unsigned int getObject() const;
	static ShaderSource load(std::string path, std::string include_indentifier = "#include");
	static void getFilePath(const std::string& fullPath, std::string& path_without_filename);

private:
	unsigned int object;
};
