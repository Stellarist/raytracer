#pragma once

#include <string>
#include <vector>
#include "RadeonRays/bvh.h"
#include "RadeonRays/bvh_translator.h"
#include "RadeonRays/split_bvh.h"
#include "Camera.hpp"
#include "Environment.hpp"
#include <Light.hpp>
#include "Material.hpp"
#include "Mesh.hpp"
#include "Renderer.hpp"
#include "Texture.hpp"

struct Indices {
	int x, y, z;
};

class Scene {
public:
	Scene();
	~Scene();

	int addMesh(const std::string& filename);
	int addTexture(const std::string& filename);
	int addMaterial(const Material& material);
	int addMeshInstance(const MeshInstance& meshInstance);
	int addLight(const Light& light);

	void addCamera(glm::vec3 eye, glm::vec3 lookat, float fov);
	void addEnvMap(const std::string& filename);

	void processScene();
	void rebuildInstances();

	RenderOptions renderOptions;
	Camera* camera;
	Environment* envMap;
	std::vector<Mesh*> meshes;
	std::vector<Indices> vertIndices;
	std::vector<glm::vec4> verticesUV_x;
	std::vector<glm::vec4> normalsUV_y;
	std::vector<glm::mat4> transforms;
	std::vector<Material> materials;
	std::vector<MeshInstance> meshInstances;
	std::vector<Light> lights;
	std::vector<Texture*> textures;
	std::vector<unsigned char> textureMapsArray;
	RadeonRays::BvhTranslator bvhTranslator;
	RadeonRays::bbox sceneBounds;

	bool dirty;
	bool initialized;
	bool instances_modified = false;
	bool env_map_modified = false;

private:
	RadeonRays::Bvh* sceneBvh;

	void createBLAS();
	void createTLAS();
};
