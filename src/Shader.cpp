#include "Shader.hpp"

#include <iostream>
#include <fstream>
#include <glad/glad.h>

Shader::Shader(const ShaderSource& source_obj, unsigned int shader_type)
{
	object = glCreateShader(shader_type);
	std::cerr << "Compiling Shader" << source_obj.path.c_str() << std::endl;

	const char* src = (const GLchar*)source_obj.src.c_str();
	glShaderSource(object, 1, &src, 0);
	glCompileShader(object);

	int success = 0;
	glGetShaderiv(object, GL_COMPILE_STATUS, &success);
	if (success == GL_FALSE) {
		std::string msg;
		int log_size = 0;
		glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_size);

		char* info = new char[log_size + 1];
		glGetShaderInfoLog(object, log_size, NULL, info);
		msg = msg + source_obj.path + "\n" + info;
		std::cerr << "Shader compilation error" << msg.c_str() << std::endl;

		object = 0;
		delete[] info;
		glDeleteShader(object);

		throw std::runtime_error(msg.c_str());
	}
}

unsigned int Shader::getObject() const
{
	return object;
}

ShaderSource Shader::load(std::string path, std::string include_indentifier)
{
	include_indentifier += ' ';
	static bool is_recursive_call = false;

	std::string full_source_code = "";
	std::ifstream file(path);

	if (!file.is_open()) {
		std::cerr << "ERROR: could not open the shader at: " << path << "\n" << std::endl;
		return ShaderSource{ full_source_code, path };
	}

	std::string lineBuffer;
	while (std::getline(file, lineBuffer)) {
		if (lineBuffer.find(include_indentifier) != lineBuffer.npos) {
			lineBuffer.erase(0, include_indentifier.size());
			std::string pathOfThisFile;
			getFilePath(path, pathOfThisFile);
			lineBuffer.insert(0, pathOfThisFile);
			is_recursive_call = true;
			full_source_code += load(lineBuffer).src;
			continue;
		}

		full_source_code += lineBuffer + '\n';
	}

	if (!is_recursive_call)
		full_source_code += '\0';
	file.close();

	return ShaderSource{ full_source_code, path };;
}

void Shader::getFilePath(const std::string& fullPath, std::string& path_without_filename)
{
	size_t found = fullPath.find_last_of("/\\");
	path_without_filename = fullPath.substr(0, found + 1);
}
