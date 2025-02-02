#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "RadeonRays/split_bvh.h"

struct MeshInstance {
    std::string name;
    int mesh_id;
    int material_id;
    glm::mat4 transform;
};

class Mesh {
public:
    Mesh();
    ~Mesh();

    void BuildBVH();
    bool LoadFromFile(const std::string& filename);

    std::vector<glm::vec4> verticesUV_x;
    std::vector<glm::vec4> normalsUV_y;

    RadeonRays::Bvh* bvh;
    std::string name;
};
