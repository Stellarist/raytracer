#include "Scene.hpp"
#include <iostream>
#include "stb/stb_image_resize.h"
#include "stb/stb_image.h"

Scene::Scene()
	: camera(nullptr), envMap(nullptr), initialized(false), dirty(true)
{
	sceneBvh = new RadeonRays::Bvh(10.0f, 64, false);
}

Scene::~Scene()
{
	for (int i = 0; i < meshes.size(); i++)
		delete meshes[i];
	meshes.clear();

	for (int i = 0; i < textures.size(); i++)
		delete textures[i];
	textures.clear();

	if (camera)
		delete camera;

	if (sceneBvh)
		delete sceneBvh;

	if (envMap)
		delete envMap;
};

void Scene::addCamera(glm::vec3 pos, glm::vec3 lookAt, float fov)
{
	delete camera;
	camera = new Camera(pos, lookAt, fov);
}

int Scene::addMesh(const std::string& filename)
{
	int id = -1;
	for (int i = 0; i < meshes.size(); i++)
		if (meshes[i]->name == filename)
			return i;

	id = meshes.size();
	Mesh* mesh = new Mesh;

	std::cerr << "Loading model" << filename.c_str() << std::endl;
	if (mesh->LoadFromFile(filename))
		meshes.push_back(mesh);
	else {
		std::cerr << "Unable to load model" << filename.c_str() << std::endl;
		delete mesh;
		id = -1;
	}
	return id;
}

int Scene::addTexture(const std::string& filename)
{
	int id = -1;
	// Check if texture was already loaded
	for (int i = 0; i < textures.size(); i++)
		if (textures[i]->name == filename)
			return i;

	id = textures.size();
	Texture* texture = new Texture;

	printf("Loading texture %s\n", filename.c_str());
	if (texture->LoadTexture(filename))
		textures.push_back(texture);
	else
	{
		printf("Unable to load texture %s\n", filename.c_str());
		delete texture;
		id = -1;
	}

	return id;
}

int Scene::addMaterial(const Material& material)
{
	int id = materials.size();
	materials.push_back(material);
	return id;
}

void Scene::addEnvMap(const std::string& filename)
{
	if (envMap)
		delete envMap;

	envMap = new Environment;
	if (envMap->loadMap(filename.c_str()))
		printf("HDR %s loaded\n", filename.c_str());
	else
	{
		printf("Unable to load HDR\n");
		delete envMap;
		envMap = nullptr;
	}
	env_map_modified = true;
	dirty = true;
}

int Scene::addMeshInstance(const MeshInstance& meshInstance)
{
	int id = meshInstances.size();
	meshInstances.push_back(meshInstance);
	return id;
}

int Scene::addLight(const Light& light)
{
	int id = lights.size();
	lights.push_back(light);
	return id;
}

void Scene::createTLAS()
{
	// Loop through all the mesh Instances and build a Top Level BVH
	std::vector<RadeonRays::bbox> bounds;
	bounds.resize(meshInstances.size());

	for (int i = 0; i < meshInstances.size(); i++)
	{
		RadeonRays::bbox bbox = meshes[meshInstances[i].mesh_id]->bvh->Bounds();
		glm::mat4 matrix = meshInstances[i].transform;

		glm::vec3 minBound = bbox.pmin;
		glm::vec3 maxBound = bbox.pmax;

		glm::vec3 right = glm::vec3(matrix[0][0], matrix[0][1], matrix[0][2]);
		glm::vec3 up = glm::vec3(matrix[1][0], matrix[1][1], matrix[1][2]);
		glm::vec3 forward = glm::vec3(matrix[2][0], matrix[2][1], matrix[2][2]);
		glm::vec3 translation = glm::vec3(matrix[3][0], matrix[3][1], matrix[3][2]);

		glm::vec3 xa = right * minBound.x;
		glm::vec3 xb = right * maxBound.x;

		glm::vec3 ya = up * minBound.y;
		glm::vec3 yb = up * maxBound.y;

		glm::vec3 za = forward * minBound.z;
		glm::vec3 zb = forward * maxBound.z;

		minBound = glm::min(xa, xb) + glm::min(ya, yb) + glm::min(za, zb) + translation;
		maxBound = glm::max(xa, xb) + glm::max(ya, yb) + glm::max(za, zb) + translation;

		RadeonRays::bbox bound;
		bound.pmin = minBound;
		bound.pmax = maxBound;

		bounds[i] = bound;
	}
	sceneBvh->Build(&bounds[0], bounds.size());
	sceneBounds = sceneBvh->Bounds();
}

void Scene::createBLAS()
{
	// Loop through all meshes and build BVHs
#pragma omp parallel for
	for (int i = 0; i < meshes.size(); i++)
	{
		printf("Building BVH for %s\n", meshes[i]->name.c_str());
		meshes[i]->BuildBVH();
	}
}

void Scene::rebuildInstances()
{
	delete sceneBvh;
	sceneBvh = new RadeonRays::Bvh(10.0f, 64, false);

	createTLAS();
	bvhTranslator.UpdateTLAS(sceneBvh, meshInstances);

	//Copy transforms
	for (int i = 0; i < meshInstances.size(); i++)
		transforms[i] = meshInstances[i].transform;

	instances_modified = true;
	dirty = true;
}

void Scene::processScene()
{
	printf("Processing scene data\n");
	createBLAS();

	printf("Building scene BVH\n");
	createTLAS();

	// Flatten BVH
	printf("Flattening BVH\n");
	bvhTranslator.Process(sceneBvh, meshes, meshInstances);

	// Copy mesh data
	int verticesCnt = 0;
	printf("Copying Mesh Data\n");
	for (int i = 0; i < meshes.size(); i++)
	{
		// Copy indices from BVH and not from Mesh. 
		// Required if splitBVH is used as a triangle can be shared by leaf nodes
		int numIndices = meshes[i]->bvh->GetNumIndices();
		const int* triIndices = meshes[i]->bvh->GetIndices();

		for (int j = 0; j < numIndices; j++)
		{
			int index = triIndices[j];
			int v1 = (index * 3 + 0) + verticesCnt;
			int v2 = (index * 3 + 1) + verticesCnt;
			int v3 = (index * 3 + 2) + verticesCnt;

			vertIndices.push_back(Indices{ v1, v2, v3 });
		}

		verticesUV_x.insert(verticesUV_x.end(), meshes[i]->verticesUV_x.begin(), meshes[i]->verticesUV_x.end());
		normalsUV_y.insert(normalsUV_y.end(), meshes[i]->normalsUV_y.begin(), meshes[i]->normalsUV_y.end());

		verticesCnt += meshes[i]->verticesUV_x.size();
	}

	// Copy transforms
	printf("Copying transforms\n");
	transforms.resize(meshInstances.size());
	for (int i = 0; i < meshInstances.size(); i++)
		transforms[i] = meshInstances[i].transform;

	// Copy textures
	if (!textures.empty())
		printf("Copying and resizing textures\n");

	int req_width = renderOptions.tex_array_width;
	int req_height = renderOptions.tex_array_height;
	int tex_bytes = req_width * req_height * 4;
	textureMapsArray.resize(tex_bytes * textures.size());

#pragma omp parallel for
	for (int i = 0; i < textures.size(); i++) {
		int tex_width = textures[i]->width;
		int texHeight = textures[i]->height;

		// Resize textures to fit 2D texture array
		if (tex_width != req_width || texHeight != req_height) {
			unsigned char* resizedTex = new unsigned char[tex_bytes];
			stbir_resize_uint8(&textures[i]->tex_data[0], tex_width, texHeight, 0, resizedTex, req_width, req_height, 0, 4);
			std::copy(resizedTex, resizedTex + tex_bytes, &textureMapsArray[i * tex_bytes]);
			delete[] resizedTex;
		}
		else
			std::copy(textures[i]->tex_data.begin(), textures[i]->tex_data.end(), &textureMapsArray[i * tex_bytes]);
	}

	// Add a default camera
	if (!camera) {
		RadeonRays::bbox bounds = sceneBvh->Bounds();
		glm::vec3 extents = bounds.extents();
		glm::vec3 center = bounds.center();
		addCamera(glm::vec3(center.x, center.y, center.z + glm::length(extents) * 2.0f), center, 45.0f);
	}

	initialized = true;
}
