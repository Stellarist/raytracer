#pragma once

#include "Program.hpp"

class Quad
{
public:
	Quad();
	void draw(Program* program);

private:
	unsigned int vao;
	unsigned int vbo;
};