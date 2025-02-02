#define TINYOBJLOADER_IMPLEMENTATION

#include "Mesh.hpp"

#include <iostream>
#include "tinyobjloader/tiny_obj_loader.h"

constexpr float M_PI = 3.14159265358979323846f;

float sphericalTheta(const glm::vec3& v)
{
	return acosf(glm::clamp(v.y, -1.f, 1.f));
}

float sphericalPhi(const glm::vec3& v)
{
	float p = atan2f(v.z, v.x);
	return (p < 0.f) ? p + 2.f * M_PI : p;
}

Mesh::Mesh()
{
	bvh = new RadeonRays::SplitBvh(2.0f, 64, 0, 0.001f, 0);
}

Mesh::~Mesh()
{
	delete bvh;
}

bool Mesh::LoadFromFile(const std::string& filename)
{
	name = filename;
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string error;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &error, filename.c_str(), 0, true)) {
		std::cerr << "Unable to load model." << std::endl;
		return false;
	}

	for (size_t s = 0; s < shapes.size(); s++) {
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			for (size_t v = 0; v < 3; v++) {
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
				tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
				tinyobj::real_t tx, ty;

				if (!attrib.texcoords.empty()) {
					tx = attrib.texcoords[2 * idx.texcoord_index + 0];
					ty = 1.0 - attrib.texcoords[2 * idx.texcoord_index + 1];
				}
				else if (v == 0)
					tx = ty = 0;
				else if (v == 1)
					tx = 0, ty = 1;
				else
					tx = ty = 1;

				verticesUV_x.push_back(glm::vec4(vx, vy, vz, tx));
				normalsUV_y.push_back(glm::vec4(nx, ny, nz, ty));
			}

			index_offset += 3;
		}
	}

	glm::vec3 center(0.0, 0.0, 0.0);

	for (int i = 0; i < verticesUV_x.size(); i++)
		center = center + glm::vec3(verticesUV_x[i]);
	center = center * static_cast<float>(1.0 / verticesUV_x.size());

	for (int i = 0; i < verticesUV_x.size(); i++)
	{
		glm::vec3 diff = glm::vec3(verticesUV_x[i]) - center;
		diff = glm::normalize(diff);
		verticesUV_x[i].w = sphericalTheta(diff) * (1.0 / M_PI);
		normalsUV_y[i].w = sphericalPhi(diff) * (1.0 / (2.0 * M_PI));
	}

	return true;
}

void Mesh::BuildBVH()
{
	const int numTris = verticesUV_x.size() / 3;
	std::vector<RadeonRays::bbox> bounds(numTris);

#pragma omp parallel for
	for (int i = 0; i < numTris; ++i) {
		const glm::vec3 v1 = glm::vec3(verticesUV_x[i * 3 + 0]);
		const glm::vec3 v2 = glm::vec3(verticesUV_x[i * 3 + 1]);
		const glm::vec3 v3 = glm::vec3(verticesUV_x[i * 3 + 2]);

		bounds[i].grow(v1);
		bounds[i].grow(v2);
		bounds[i].grow(v3);
	}

	bvh->Build(&bounds[0], numTris);
}
