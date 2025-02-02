#pragma once

#include <glm/glm.hpp>

class Camera {
public:
	Camera(glm::vec3 eye, glm::vec3 lookat, float fov);
	Camera(const Camera& other);
	Camera& operator = (const Camera& other);

	void strafe(float dx, float dy);
	void setFov(float val);
	void setRadius(float dr);
	void OffsetOrientation(float dx, float dy);
	void ComputeViewProjectionMatrix(float* view, float* projection, float ratio);

	glm::vec3 position;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 forward;
	glm::vec3 world_up;
	glm::vec3 pivot;

	float focal_dist;
	float aperture;
	float fov;
	float pitch;
	float yaw;
	float radius;
	bool is_moving;

	void updateCamera();
};