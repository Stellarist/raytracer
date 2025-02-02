#include "Camera.hpp"

#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

Camera::Camera(glm::vec3 eye, glm::vec3 lookat, float fov)
	: position(eye), pivot(lookat), world_up(0, 1, 0),
	focal_dist(0.1f), aperture(0.0), fov(glm::radians(fov))
{
	glm::vec3 dir = glm::normalize(lookat - eye);
	pitch = glm::degrees(asin(dir.y));
	yaw = glm::degrees(atan2(dir.z, dir.x));
	radius = glm::distance(eye, lookat);
	updateCamera();
}

Camera::Camera(const Camera& other)
{
	*this = other;
}

Camera& Camera::operator = (const Camera& other)
{
	if (this == &other)
		return *this;

	std::memcpy(this, &other, sizeof(Camera));

	return *this;
}

void Camera::OffsetOrientation(float dx, float dy)
{
	pitch -= dy;
	yaw += dx;
	updateCamera();
}

void Camera::strafe(float dx, float dy)
{
	glm::vec3 translation = right * -dx + up * dy;
	pivot = pivot + translation;
	updateCamera();
}

void Camera::setFov(float val)
{
	fov = glm::radians(val);
}

void Camera::setRadius(float dr)
{
	radius += dr;
	updateCamera();
}

void Camera::ComputeViewProjectionMatrix(float* view, float* projection, float ratio)
{
	glm::mat4 view_mat = glm::lookAt(position, position + forward, up);
	glm::mat4 perspective_mat = glm::perspective(glm::degrees((1.f / ratio) * tanf(fov / 2.f)), ratio, 0.1f, 1000.f);

	memcpy(view, glm::value_ptr(view_mat), sizeof(float) * 16);
	memcpy(projection, glm::value_ptr(perspective_mat), sizeof(float) * 16);
}

void Camera::updateCamera()
{
	forward = glm::normalize(glm::vec3(
		cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
		sin(glm::radians(pitch)),
		sin(glm::radians(yaw)) * cos(glm::radians(pitch))
	));
	position = pivot + (forward * -1.0f) * radius;

	right = glm::normalize(glm::cross(forward, world_up));
	up = glm::normalize(glm::cross(right, forward));
}
