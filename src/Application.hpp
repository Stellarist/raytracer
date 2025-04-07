#pragma once

#include <optional>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

struct QueueFamilyIndices {
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;

	bool isComplete() {
		return graphics_family.has_value()
			&& present_family.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR        capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR>   present_modes;
};

class Application {
private:
	GLFWwindow* window;
	VkInstance                 instance;
	VkDebugUtilsMessengerEXT   debug_messenger;
	VkPhysicalDevice           gpu;
	VkDevice                   device;
	VkQueue                    graphics_queue;
	VkQueue                    present_queue;
	VkSurfaceKHR               surface;
	VkSwapchainKHR             swap_chain;
	std::vector<VkImage>       swap_chain_images;
	VkFormat                   swap_chain_image_format;
	VkExtent2D                 swap_chain_extent;
	std::vector<VkImageView>   swap_chain_image_views;
	VkRenderPass               render_pass;
	VkPipelineLayout           pipeline_layout;
	VkPipeline                 graphics_pipeline;
	std::vector<VkFramebuffer> swap_chain_framebuffers;
	VkCommandPool              command_pool;
	VkCommandBuffer            command_buffer;
	VkSemaphore                image_available_semaphore;
	VkSemaphore                render_finished_semaphore;
	VkFence                    in_flight_fence;

	void initWindow();
	void initVulkan();
	void loop();
	void draw();
	void release();
	void createInstance();
	void createSurface();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createSwapChain();
	void createImageViews();
	void createRenderPass();
	void createGraphicsPipeline();
	void createFramebuffers();
	void createCommandPool();
	void createCommandBuffer();
	void createSyncObjects();

	bool isDeviceSuitable(const VkPhysicalDevice& device);
	bool checkValidationLayerSupport();
	bool checkDeviceExtensionSupport(const VkPhysicalDevice& device);
	void setupDebugMessenger();
	void recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index);
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info);
	VkShaderModule createShaderModule(const std::vector<char>& code);
	std::vector<const char*> getRequiredExtensions();
	SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice& device);
	QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice& device);
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats);
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available_present_modes);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
	VkResult createDebugUtilsMessengerEXT(
		VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
		const VkAllocationCallbacks* p_allocator,
		VkDebugUtilsMessengerEXT* p_debug_messenger);
	void destroyDebugUtilsMessengerEXT(
		VkInstance instance,
		VkDebugUtilsMessengerEXT debug_messenger,
		const VkAllocationCallbacks* p_allocator);

	static std::vector<char> readFile(const std::string& filename);
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type,
		const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
		void* p_user_data);

public:
	Application() = default;
	~Application() = default;

	void run();
};