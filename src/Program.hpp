#pragma once

#include <vector>
#include "Shader.hpp"

class Program {
public:
	Program(const std::vector<Shader> shaders);
	~Program();
	void use();
	void stopusing();
	unsigned int getObject();

private:
	unsigned int object;
};
