#include "Program.hpp"

#include <iostream>
#include <stdexcept>
#include <glad/glad.h>

Program::Program(const std::vector<Shader> shaders)
{
	object = glCreateProgram();
	for (unsigned int i = 0; i < shaders.size(); i++)
		glAttachShader(object, shaders[i].getObject());
	glLinkProgram(object);

	for (unsigned i = 0; i < shaders.size(); i++)
		glDetachShader(object, shaders[i].getObject());

	int success = 0;
	glGetProgramiv(object, GL_LINK_STATUS, &success);
	if (success == GL_FALSE) {
		std::string msg("Error while linking program\n");
		int log_size = 0;
		glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_size);

		char* info = new char[log_size + 1];
		glGetShaderInfoLog(object, log_size, NULL, info);
		msg += info;
		std::cerr << "Shader program error" << msg.c_str() << std::endl;

		object = 0;
		delete[] info;
		glDeleteProgram(object);

		throw std::runtime_error(msg.c_str());
	}
}

Program::~Program()
{
	glDeleteProgram(object);
}

void Program::use()
{
	glUseProgram(object);
}

void Program::stopusing()
{
	glUseProgram(0);
}

unsigned int Program::getObject()
{
	return object;
}
