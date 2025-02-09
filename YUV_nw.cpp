/*
* Vulkan Example - Basic indexed triangle rendering
*
* Note:
*	This is a "pedal to the metal" example to show off how to get Vulkan up and displaying something
*	Contrary to the other examples, this one won't make use of helper functions or initializers
*	Except in a few cases (swap chain setup e.g.)
*
* Copyright (C) 2016-2024 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <vector>
#include <exception>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"

#include <mutex>

class DebugLogger {
private:
	static std::mutex mtx;
	static const int MAX_BUFFER = 1024;
	static HWND debugWindow;
	static std::vector<std::string> logHistory;
	static const size_t MAX_HISTORY = 1000;

public:
	static void Initialize(HWND hwnd) {
		debugWindow = hwnd;
	}

	template<typename... Args>
	static void Log(const char* format, Args... args) {
		std::lock_guard<std::mutex> lock(mtx);

		char buffer[MAX_BUFFER];
		snprintf(buffer, MAX_BUFFER, format, args...);

		// Add newline
		std::string message(buffer);
		message += "\n";

		// Store in history
		logHistory.push_back(message);
		if (logHistory.size() > MAX_HISTORY) {
			logHistory.erase(logHistory.begin());
		}

		// Output to debug window
		OutputDebugStringA(message.c_str());

		// Update window if available
		if (debugWindow && IsWindow(debugWindow)) {
			// Combine all messages
			std::string allMessages;
			for (const auto& msg : logHistory) {
				allMessages += msg;
			}

			// Update window text
			SetWindowTextA(debugWindow, allMessages.c_str());

			// Scroll to bottom
			SendMessage(debugWindow, EM_SETSEL, 0, -1);
			SendMessage(debugWindow, EM_SETSEL, -1, -1);
			SendMessage(debugWindow, EM_SCROLLCARET, 0, 0);
		}
	}

	static void Clear() {
		std::lock_guard<std::mutex> lock(mtx);
		logHistory.clear();
		if (debugWindow && IsWindow(debugWindow)) {
			SetWindowTextA(debugWindow, "");
		}
	}
};

std::mutex DebugLogger::mtx;
HWND DebugLogger::debugWindow = NULL;
std::vector<std::string> DebugLogger::logHistory;

#define DEBUG_LOG(...) DebugLogger::Log(__VA_ARGS__)
#define DEBUG_CLEAR() DebugLogger::Clear()
#define DEBUG_INIT(hwnd) DebugLogger::Initialize(hwnd)

// We want to keep GPU and CPU busy. To do that we may start building a new command buffer while the previous one is still being executed
// This number defines how many frames may be worked on simultaneously at once
// Increasing this number may improve performance but will also introduce additional latency
#define MAX_CONCURRENT_FRAMES 2

class VulkanExample : public VulkanExampleBase
{
private:
	// Add new members
	struct {
		VkPipeline pipeline{ VK_NULL_HANDLE };        // Pipeline for pass-through rendering
		VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
		VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
		VkDescriptorPool descriptorPool{ VK_NULL_HANDLE };
		std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets{};
	} passthrough;

	// Add new members for pass-through rendering
	struct PassthroughStage {
		VkPipeline pipeline{ VK_NULL_HANDLE };
		VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
		VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
		VkDescriptorPool descriptorPool{ VK_NULL_HANDLE };
		std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets{};
		bool isYuvInput{ false };
		bool isYuvOutput{ false };
	};

	PassthroughStage rt1ToRt2Stage;
	PassthroughStage rt2ToRt3Stage;
	PassthroughStage rt3ToSwapchainStage;

public:
	// Vertex layout used in this example
	struct Vertex {
		float position[3];
		float color[3];
	};

	// Vertex buffer and attributes
	struct {
		VkDeviceMemory memory{ VK_NULL_HANDLE }; // Handle to the device memory for this buffer
		VkBuffer buffer{ VK_NULL_HANDLE };		 // Handle to the Vulkan buffer object that the memory is bound to
	} vertices;

	// Index buffer
	struct {
		VkDeviceMemory memory{ VK_NULL_HANDLE };
		VkBuffer buffer{ VK_NULL_HANDLE };
		uint32_t count{ 0 };
	} indices;

	struct RenderTarget {
		VkImage image{ VK_NULL_HANDLE };
		VkDeviceMemory memory{ VK_NULL_HANDLE };
		std::vector<VkImageView> views;  // For multi-plane formats
		VkSampler sampler{ VK_NULL_HANDLE };
		VkFormat format{ VK_FORMAT_R8G8B8A8_UNORM };
		uint32_t width{ 800 };
		uint32_t height{ 600 };
		bool isYuv{ false };
	};

	struct {
		VkBuffer buffer{ VK_NULL_HANDLE };
		VkDeviceMemory memory{ VK_NULL_HANDLE };
	} fullscreenQuad;

	// NEW: Second render target for intermediate copy
	RenderTarget renderTarget1{};  // Original render target
	RenderTarget renderTarget2{};  // New render target for intermediate copy
	VkFramebuffer renderTarget1Framebuffer{ VK_NULL_HANDLE };
	VkFramebuffer renderTarget2Framebuffer{ VK_NULL_HANDLE };
	// Add new render target
	RenderTarget renderTarget3{};  // New render target for second copy stage
	VkFramebuffer renderTarget3Framebuffer{ VK_NULL_HANDLE };

	// Add to class VulkanExample private members
	struct StageSyncObjects {
		std::array<VkCommandBuffer, MAX_CONCURRENT_FRAMES> commandBuffers{};
		std::array<VkSemaphore, MAX_CONCURRENT_FRAMES> completeSemaphores{};
		std::array<VkFence, MAX_CONCURRENT_FRAMES> fences{};
	};

	// One set of sync objects for each stage
	StageSyncObjects renderStage;      // For initial render to RT1
	StageSyncObjects firstCopyStage;   // For copy from RT1 to RT2
	StageSyncObjects secondCopyStage;     // RT2 to RT3  <-- New
	StageSyncObjects finalCopyStage;   // For copy from RT2 to swapchain

	// Uniform buffer block object
	struct UniformBuffer {
		VkDeviceMemory memory{ VK_NULL_HANDLE };
		VkBuffer buffer{ VK_NULL_HANDLE };
		// The descriptor set stores the resources bound to the binding points in a shader
		// It connects the binding points of the different shaders with the buffers and images used for those bindings
		VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
		// We keep a pointer to the mapped buffer, so we can easily update it's contents via a memcpy
		uint8_t* mapped{ nullptr };
	};
	// We use one UBO per frame, so we can have a frame overlap and make sure that uniforms aren't updated while still in use
	std::array<UniformBuffer, MAX_CONCURRENT_FRAMES> uniformBuffers;

	// For simplicity we use the same uniform block layout as in the shader:
	//
	//	layout(set = 0, binding = 0) uniform UBO
	//	{
	//		mat4 projectionMatrix;
	//		mat4 modelMatrix;
	//		mat4 viewMatrix;
	//	} ubo;
	//
	// This way we can just memcopy the ubo data to the ubo
	// Note: You should use data types that align with the GPU in order to avoid manual padding (vec4, mat4)
	struct ShaderData {
		glm::mat4 projectionMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	};

	// The pipeline layout is used by a pipeline to access the descriptor sets
	// It defines interface (without binding any actual data) between the shader stages used by the pipeline and the shader resources
	// A pipeline layout can be shared among multiple pipelines as long as their interfaces match
	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };

	// Pipelines (often called "pipeline state objects") are used to bake all states that affect a pipeline
	// While in OpenGL every state can be changed at (almost) any time, Vulkan requires to layout the graphics (and compute) pipeline states upfront
	// So for each combination of non-dynamic pipeline states you need a new pipeline (there are a few exceptions to this not discussed here)
	// Even though this adds a new dimension of planning ahead, it's a great opportunity for performance optimizations by the driver
	VkPipeline pipeline{ VK_NULL_HANDLE };

	// The descriptor set layout describes the shader binding layout (without actually referencing descriptor)
	// Like the pipeline layout it's pretty much a blueprint and can be used with different descriptor sets as long as their layout matches
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };

	// Synchronization primitives
	// Synchronization is an important concept of Vulkan that OpenGL mostly hid away. Getting this right is crucial to using Vulkan.

	// Semaphores are used to coordinate operations within the graphics queue and ensure correct command ordering
	std::array<VkSemaphore, MAX_CONCURRENT_FRAMES> presentCompleteSemaphores{};
	std::array<VkSemaphore, MAX_CONCURRENT_FRAMES> renderCompleteSemaphores{};

	VkCommandPool commandPool{ VK_NULL_HANDLE };
	std::array<VkCommandBuffer, MAX_CONCURRENT_FRAMES> commandBuffers{};
	std::array<VkFence, MAX_CONCURRENT_FRAMES> waitFences{};

	// To select the correct sync objects, we need to keep track of the current frame
	uint32_t currentFrame{ 0 };

	VulkanExample() : VulkanExampleBase()
	{
		title = "Vulkan Example - Basic indexed triangle";
		// To keep things simple, we don't use the UI overlay from the framework
		settings.overlay = false;
		// Setup a default look-at camera
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
		camera.setRotation(glm::vec3(0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 1.0f, 256.0f);
		// Values not set here are initialized in the base class constructor
	}

	void cleanupStageSyncObjects(StageSyncObjects& stage)
	{
		// Note: Command buffers are freed when command pool is destroyed
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			vkDestroySemaphore(device, stage.completeSemaphores[i], nullptr);
			vkDestroyFence(device, stage.fences[i], nullptr);
		}
	}

	void cleanupPassthroughStage(PassthroughStage& stage) {
		vkDestroyPipeline(device, stage.pipeline, nullptr);
		vkDestroyPipelineLayout(device, stage.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, stage.descriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, stage.descriptorPool, nullptr);
	}

	void cleanupRenderTarget(RenderTarget& rt) {
		for (auto& view : rt.views) {
			vkDestroyImageView(device, view, nullptr);
		}
		rt.views.clear();

		if (rt.sampler != VK_NULL_HANDLE) {
			vkDestroySampler(device, rt.sampler, nullptr);
			rt.sampler = VK_NULL_HANDLE;
		}
		if (rt.image != VK_NULL_HANDLE) {
			vkDestroyImage(device, rt.image, nullptr);
			rt.image = VK_NULL_HANDLE;
		}
		if (rt.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, rt.memory, nullptr);
			rt.memory = VK_NULL_HANDLE;
		}
	}

	~VulkanExample() {
		// Clean up used Vulkan resources
		// Note: Inherited destructor cleans up resources stored in base class

		// Clean up YUV UBO
		if (yuvUbo.mapped) {
			vkUnmapMemory(device, yuvUbo.memory);
		}
		if (yuvUbo.buffer != VK_NULL_HANDLE) {
			vkDestroyBuffer(device, yuvUbo.buffer, nullptr);
		}
		if (yuvUbo.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, yuvUbo.memory, nullptr);
		}

		// Clean up render targets
		cleanupRenderTarget(renderTarget1);
		cleanupRenderTarget(renderTarget2);
		cleanupRenderTarget(renderTarget3);

		// Clean up framebuffers
		vkDestroyFramebuffer(device, renderTarget1Framebuffer, nullptr);
		vkDestroyFramebuffer(device, renderTarget2Framebuffer, nullptr);
		vkDestroyFramebuffer(device, renderTarget3Framebuffer, nullptr);

		// Clean up synchronization objects
		cleanupStageSyncObjects(renderStage);
		cleanupStageSyncObjects(firstCopyStage);
		cleanupStageSyncObjects(secondCopyStage);
		cleanupStageSyncObjects(finalCopyStage);

		// Clean up pipeline stages
		cleanupPassthroughStage(rt1ToRt2Stage);
		cleanupPassthroughStage(rt2ToRt3Stage);
		cleanupPassthroughStage(rt3ToSwapchainStage);

		// Clean up command pool
		vkDestroyCommandPool(device, commandPool, nullptr);
	}

	void createPassthroughDescriptorSetLayout(PassthroughStage& stage) {
		std::vector<VkDescriptorSetLayoutBinding> bindings;

		if (stage.isYuvInput) {
			// YUV input needs two samplers (Y and UV planes)
			VkDescriptorSetLayoutBinding yBinding{};
			yBinding.binding = 0;
			yBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			yBinding.descriptorCount = 1;
			yBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			bindings.push_back(yBinding);

			VkDescriptorSetLayoutBinding uvBinding = yBinding;
			uvBinding.binding = 1;
			bindings.push_back(uvBinding);
		}
		else {
			// RGB input needs only one sampler
			VkDescriptorSetLayoutBinding samplerBinding{};
			samplerBinding.binding = 0;
			samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			samplerBinding.descriptorCount = 1;
			samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			bindings.push_back(samplerBinding);
		}

		// Add UBO binding for height information (needed for YUV conversion)
		VkDescriptorSetLayoutBinding uboBinding{};
		uboBinding.binding = bindings.size();  // Next available binding
		uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboBinding.descriptorCount = 1;
		uboBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings.push_back(uboBinding);

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &stage.descriptorSetLayout));
	}

	void createPassthroughDescriptorPool(PassthroughStage& stage) {
		std::vector<VkDescriptorPoolSize> poolSizes;

		// Sampler pool sizes
		VkDescriptorPoolSize samplerPoolSize{};
		samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerPoolSize.descriptorCount = MAX_CONCURRENT_FRAMES * (stage.isYuvInput ? 2 : 1);
		poolSizes.push_back(samplerPoolSize);

		// UBO pool size
		VkDescriptorPoolSize uboPoolSize{};
		uboPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboPoolSize.descriptorCount = MAX_CONCURRENT_FRAMES;
		poolSizes.push_back(uboPoolSize);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.maxSets = MAX_CONCURRENT_FRAMES;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &stage.descriptorPool));
	}

	struct YuvUBO {
		int32_t height;
		float padding[3];  // Keep alignment requirements
	};

	struct {
		VkBuffer buffer;
		VkDeviceMemory memory;
		void* mapped;
	} yuvUbo;

	void createYuvUbo() {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(YuvUBO);
		bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &yuvUbo.buffer));

		VkMemoryRequirements memReqs;
		vkGetBufferMemoryRequirements(device, yuvUbo.buffer, &memReqs);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memReqs.size;
		allocInfo.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &yuvUbo.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, yuvUbo.buffer, yuvUbo.memory, 0));
		VK_CHECK_RESULT(vkMapMemory(device, yuvUbo.memory, 0, sizeof(YuvUBO), 0, &yuvUbo.mapped));
	}

	void createCommandPool()
	{
		// Create command pool
		VkCommandPoolCreateInfo commandPoolCI{};
		commandPoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolCI.queueFamilyIndex = swapChain.queueNodeIndex;
		commandPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCI, nullptr, &commandPool));
	}


	void createStageSyncObjects(StageSyncObjects& stage, const char* debugName)
	{
		// Create semaphores and fences
		VkSemaphoreCreateInfo semaphoreCI{};
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceCI{};
		fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Create in signaled state

		// Create for each frame in flight
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			// Create and check semaphore
			VkResult semResult = vkCreateSemaphore(device, &semaphoreCI, nullptr, &stage.completeSemaphores[i]);
			if (semResult != VK_SUCCESS) {
				throw std::runtime_error(std::string("Failed to create semaphore for stage ") +
					debugName + " frame " + std::to_string(i));
			}

			// Create and check fence
			VkResult fenceResult = vkCreateFence(device, &fenceCI, nullptr, &stage.fences[i]);
			if (fenceResult != VK_SUCCESS) {
				throw std::runtime_error(std::string("Failed to create fence for stage ") +
					debugName + " frame " + std::to_string(i));
			}

			// Verify creation
			if (stage.completeSemaphores[i] == VK_NULL_HANDLE || stage.fences[i] == VK_NULL_HANDLE) {
				throw std::runtime_error(std::string("Sync objects are null for stage ") +
					debugName + " frame " + std::to_string(i));
			}
		}
	}

	// Add this verification function to check sync objects before use
	void verifySyncObjects()
	{
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			// Check render stage
			if (renderStage.fences[i] == VK_NULL_HANDLE) {
				throw std::runtime_error("Render stage fence is null for frame " + std::to_string(i));
			}

			// Check first copy stage
			if (firstCopyStage.fences[i] == VK_NULL_HANDLE) {
				throw std::runtime_error("First copy stage fence is null for frame " + std::to_string(i));
			}

			// Check second copy stage
			if (secondCopyStage.fences[i] == VK_NULL_HANDLE) {
				throw std::runtime_error("Second copy stage fence is null for frame " + std::to_string(i));
			}

			// Check final copy stage
			if (finalCopyStage.fences[i] == VK_NULL_HANDLE) {
				throw std::runtime_error("Final copy stage fence is null for frame " + std::to_string(i));
			}
		}
	}

	void createSynchronizationPrimitives()
	{
		// Create sync objects for all stages
		createStageSyncObjects(renderStage, "render");
		createStageSyncObjects(firstCopyStage, "firstCopy");
		createStageSyncObjects(secondCopyStage, "secondCopy");  // This was missing
		createStageSyncObjects(finalCopyStage, "finalCopy");

		// Create present semaphores
		VkSemaphoreCreateInfo semaphoreCI{};
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &presentCompleteSemaphores[i]));
		}
	}

	void createRenderTarget(RenderTarget& rt, VkFramebuffer& framebuffer, bool isYuv = false) {
		rt.isYuv = isYuv;
		rt.views.clear();

		// Image creation
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = isYuv ? VK_FORMAT_G8_B8R8_2PLANE_420_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
		imageInfo.extent = { rt.width, rt.height, 1 };
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		if (isYuv) {
			imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			// Store the format for later verification
			rt.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		}
		else {
			rt.format = VK_FORMAT_R8G8B8A8_UNORM;
		}

		DEBUG_LOG("Creating image with format: %d for %s target", imageInfo.format, isYuv ? "YUV" : "RGB");

		VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &rt.image));

		// Memory allocation
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, rt.image, &memReqs);

		VkMemoryAllocateInfo memAlloc{};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &rt.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, rt.image, rt.memory, 0));

		if (isYuv) {
			// Create separate view for Y plane (R8)
			VkImageViewCreateInfo yViewInfo{};
			yViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			yViewInfo.image = rt.image;
			yViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			yViewInfo.format = VK_FORMAT_R8_UNORM;
			yViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
			yViewInfo.subresourceRange.baseMipLevel = 0;
			yViewInfo.subresourceRange.levelCount = 1;
			yViewInfo.subresourceRange.baseArrayLayer = 0;
			yViewInfo.subresourceRange.layerCount = 1;

			VkImageView yView;
			VK_CHECK_RESULT(vkCreateImageView(device, &yViewInfo, nullptr, &yView));
			rt.views.push_back(yView);

			// Create view for UV plane (R8G8)
			VkImageViewCreateInfo uvViewInfo = yViewInfo;
			uvViewInfo.format = VK_FORMAT_R8G8_UNORM;
			uvViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;

			VkImageView uvView;
			VK_CHECK_RESULT(vkCreateImageView(device, &uvViewInfo, nullptr, &uvView));
			rt.views.push_back(uvView);

			// Create combined view
			VkImageViewCreateInfo combinedViewInfo = yViewInfo;
			combinedViewInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
			combinedViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

			VkImageView combinedView;
			VK_CHECK_RESULT(vkCreateImageView(device, &combinedViewInfo, nullptr, &combinedView));
			rt.views.push_back(combinedView);

			DEBUG_LOG("Created YUV views: Y(%p), UV(%p), Combined(%p)",
				(void*)rt.views[0], (void*)rt.views[1], (void*)rt.views[2]);
		}
		else {
			// Standard RGBA view creation
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = rt.image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = rt.format;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;

			VkImageView view;
			VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &view));
			rt.views.push_back(view);

			DEBUG_LOG("Created RGB view: %p", (void*)rt.views[0]);
		}

		// Create sampler
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

		VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &rt.sampler));

		// Create framebuffer
		std::array<VkImageView, 2> attachments = {
			rt.views.back(),  // Use the last view (combined view for YUV, only view for RGB)
			depthStencil.view
		};

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = rt.width;
		framebufferInfo.height = rt.height;
		framebufferInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer));
	}

	void createRenderTarget1(RenderTarget& rt, VkFramebuffer& framebuffer, bool isYuv = false) {
		rt.isYuv = isYuv;
		rt.views.clear();

		// Image creation
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = isYuv ? VK_FORMAT_G8_B8R8_2PLANE_420_UNORM : rt.format;
		rt.format = imageInfo.format;
		imageInfo.extent = { rt.width, rt.height, 1 };
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		if (isYuv) {
			imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &rt.image));

		// Memory allocation
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, rt.image, &memReqs);

		VkMemoryAllocateInfo memAlloc{};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &rt.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, rt.image, rt.memory, 0));

		// Create image views
		if (isYuv) {
			// First create the plane views
			rt.views.resize(3);  // 2 plane views + 1 combined view

			// Y plane view (plane 0)
			VkImageViewCreateInfo yViewInfo{};
			yViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			yViewInfo.image = rt.image;
			yViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			yViewInfo.format = VK_FORMAT_R8_UNORM;  // Y plane format
			yViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
			yViewInfo.subresourceRange.baseMipLevel = 0;
			yViewInfo.subresourceRange.levelCount = 1;
			yViewInfo.subresourceRange.baseArrayLayer = 0;
			yViewInfo.subresourceRange.layerCount = 1;
			VK_CHECK_RESULT(vkCreateImageView(device, &yViewInfo, nullptr, &rt.views[0]));

			// UV plane view (plane 1)
			VkImageViewCreateInfo uvViewInfo = yViewInfo;
			uvViewInfo.format = VK_FORMAT_R8G8_UNORM;  // UV plane format
			uvViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
			VK_CHECK_RESULT(vkCreateImageView(device, &uvViewInfo, nullptr, &rt.views[1]));

			// Create combined view for framebuffer
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = rt.image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = imageInfo.format;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;
			VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &rt.views[2]));
		}
		else {
			// Standard RGBA view creation
			rt.views.resize(1);
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = rt.image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = rt.format;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;
			VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &rt.views[0]));
		}

		// Create sampler
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &rt.sampler));

		// Create framebuffer using the last view (combined view for YUV, only view for RGB)
		std::array<VkImageView, 2> attachments = {
			rt.views.back(),  // Use the last view (combined view for YUV, only view for RGB)
			depthStencil.view
		};

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = rt.width;
		framebufferInfo.height = rt.height;
		framebufferInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer));

		if (isYuv) {
			// Print size of views after creation for debugging
			DEBUG_LOG("Created YUV RT with views: %d", rt.views.size());
		}
		else {
			DEBUG_LOG("Created RGB RT with views: %d", rt.views.size());
		}
	}

	// Called from prepare()
	void createRenderTargets()
	{
		createRenderTarget(renderTarget1, renderTarget1Framebuffer);
		createRenderTarget(renderTarget2, renderTarget2Framebuffer);
		createRenderTarget(renderTarget3, renderTarget3Framebuffer);  // New
	}

	// This function is used to request a device memory type that supports all the property flags we request (e.g. device local, host visible)
	// Upon success it will return the index of the memory type that fits our requested memory properties
	// This is necessary as implementations can offer an arbitrary number of memory types with different
	// memory properties.
	// You can check https://vulkan.gpuinfo.org/ for details on different memory configurations
	uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties)
	{
		// Iterate over all memory types available for the device used in this example
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
		{
			if ((typeBits & 1) == 1)
			{
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}
			typeBits >>= 1;
		}

		throw "Could not find a suitable memory type!";
	}

	// Create the per-frame (in flight) Vulkan synchronization primitives used in this example
	void createSynchronizationPrimitives1()
	{
		// Semaphores are used for correct command ordering within a queue
		VkSemaphoreCreateInfo semaphoreCI{};
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		// Fences are used to check draw command buffer completion on the host
		VkFenceCreateInfo fenceCI{};
		fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		// Create the fences in signaled state (so we don't wait on first render of each command buffer)
		fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			// Semaphore used to ensure that image presentation is complete before starting to submit again
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &presentCompleteSemaphores[i]));
			// Semaphore used to ensure that all commands submitted have been finished before submitting the image to the queue
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &renderCompleteSemaphores[i]));

			// Fence used to ensure that command buffer has completed exection before using it again
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &waitFences[i]));
		}
	}

	void createCommandBuffers()
	{



		VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.commandPool = commandPool;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandBufferCount = MAX_CONCURRENT_FRAMES;

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, renderStage.commandBuffers.data()));
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, firstCopyStage.commandBuffers.data()));
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, secondCopyStage.commandBuffers.data()));  // New
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, finalCopyStage.commandBuffers.data()));
	}

	// Prepare vertex and index buffers for an indexed triangle
	// Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
	void createVertexBuffer()
	{
		// A note on memory management in Vulkan in general:
		//	This is a very complex topic and while it's fine for an example application to small individual memory allocations that is not
		//	what should be done a real-world application, where you should allocate large chunks of memory at once instead.

		// Setup vertices
		std::vector<Vertex> vertexBuffer{
			{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
			{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
			{ {  0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
		};
		uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);

		// Setup indices
		std::vector<uint32_t> indexBuffer{ 0, 1, 2 };
		indices.count = static_cast<uint32_t>(indexBuffer.size());
		uint32_t indexBufferSize = indices.count * sizeof(uint32_t);

		VkMemoryAllocateInfo memAlloc{};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		VkMemoryRequirements memReqs;

		// Static data like vertex and index buffer should be stored on the device memory for optimal (and fastest) access by the GPU
		//
		// To achieve this we use so-called "staging buffers" :
		// - Create a buffer that's visible to the host (and can be mapped)
		// - Copy the data to this buffer
		// - Create another buffer that's local on the device (VRAM) with the same size
		// - Copy the data from the host to the device using a command buffer
		// - Delete the host visible (staging) buffer
		// - Use the device local buffers for rendering
		//
		// Note: On unified memory architectures where host (CPU) and GPU share the same memory, staging is not necessary
		// To keep this sample easy to follow, there is no check for that in place

		struct StagingBuffer {
			VkDeviceMemory memory;
			VkBuffer buffer;
		};

		struct {
			StagingBuffer vertices;
			StagingBuffer indices;
		} stagingBuffers{};

		void* data;

		// Vertex buffer
		VkBufferCreateInfo vertexBufferInfoCI{};
		vertexBufferInfoCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		vertexBufferInfoCI.size = vertexBufferSize;
		// Buffer is used as the copy source
		vertexBufferInfoCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Create a host-visible buffer to copy the vertex data to (staging buffer)
		VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferInfoCI, nullptr, &stagingBuffers.vertices.buffer));
		vkGetBufferMemoryRequirements(device, stagingBuffers.vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Request a host visible memory type that can be used to copy our data to
		// Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &stagingBuffers.vertices.memory));
		// Map and copy
		VK_CHECK_RESULT(vkMapMemory(device, stagingBuffers.vertices.memory, 0, memAlloc.allocationSize, 0, &data));
		memcpy(data, vertexBuffer.data(), vertexBufferSize);
		vkUnmapMemory(device, stagingBuffers.vertices.memory);
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0));

		// Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
		vertexBufferInfoCI.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferInfoCI, nullptr, &vertices.buffer));
		vkGetBufferMemoryRequirements(device, vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &vertices.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, vertices.buffer, vertices.memory, 0));

		// Index buffer
		VkBufferCreateInfo indexbufferCI{};
		indexbufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		indexbufferCI.size = indexBufferSize;
		indexbufferCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Copy index data to a buffer visible to the host (staging buffer)
		VK_CHECK_RESULT(vkCreateBuffer(device, &indexbufferCI, nullptr, &stagingBuffers.indices.buffer));
		vkGetBufferMemoryRequirements(device, stagingBuffers.indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &stagingBuffers.indices.memory));
		VK_CHECK_RESULT(vkMapMemory(device, stagingBuffers.indices.memory, 0, indexBufferSize, 0, &data));
		memcpy(data, indexBuffer.data(), indexBufferSize);
		vkUnmapMemory(device, stagingBuffers.indices.memory);
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0));

		// Create destination buffer with device only visibility
		indexbufferCI.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VK_CHECK_RESULT(vkCreateBuffer(device, &indexbufferCI, nullptr, &indices.buffer));
		vkGetBufferMemoryRequirements(device, indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &indices.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, indices.buffer, indices.memory, 0));

		// Buffer copies have to be submitted to a queue, so we need a command buffer for them
		// Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies
		VkCommandBuffer copyCmd;

		VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.commandPool = commandPool;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandBufferCount = 1;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));
		// Put buffer region copies into command buffer
		VkBufferCopy copyRegion{};
		// Vertex buffer
		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, vertices.buffer, 1, &copyRegion);
		// Index buffer
		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, indices.buffer, 1, &copyRegion);
		VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

		// Submit the command buffer to the queue to finish the copy
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &copyCmd;

		// Create fence to ensure that the command buffer has finished executing
		VkFenceCreateInfo fenceCI{};
		fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCI.flags = 0;
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &fence));

		// Submit to the queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		// Wait for the fence to signal that command buffer has finished executing
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));

		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, commandPool, 1, &copyCmd);

		// Destroy staging buffers
		// Note: Staging buffer must not be deleted before the copies have been submitted and executed
		vkDestroyBuffer(device, stagingBuffers.vertices.buffer, nullptr);
		vkFreeMemory(device, stagingBuffers.vertices.memory, nullptr);
		vkDestroyBuffer(device, stagingBuffers.indices.buffer, nullptr);
		vkFreeMemory(device, stagingBuffers.indices.memory, nullptr);
	}

	// Descriptors are allocated from a pool, that tells the implementation how many and what types of descriptors we are going to use (at maximum)
	void createDescriptorPool()
	{
		// We need to tell the API the number of max. requested descriptors per type
		VkDescriptorPoolSize descriptorTypeCounts[1]{};
		// This example only one descriptor type (uniform buffer)
		descriptorTypeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		// We have one buffer (and as such descriptor) per frame
		descriptorTypeCounts[0].descriptorCount = MAX_CONCURRENT_FRAMES;
		// For additional types you need to add new entries in the type count list
		// E.g. for two combined image samplers :
		// typeCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		// typeCounts[1].descriptorCount = 2;

		// Create the global descriptor pool
		// All descriptors used in this example are allocated from this pool
		VkDescriptorPoolCreateInfo descriptorPoolCI{};
		descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCI.pNext = nullptr;
		descriptorPoolCI.poolSizeCount = 1;
		descriptorPoolCI.pPoolSizes = descriptorTypeCounts;
		// Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
		// Our sample will create one set per uniform buffer per frame
		descriptorPoolCI.maxSets = MAX_CONCURRENT_FRAMES;
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorPool));
	}

	// Descriptor set layouts define the interface between our application and the shader
	// Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
	// So every shader binding should map to one descriptor set layout binding
	void createDescriptorSetLayout()
	{
		// Binding 0: Uniform buffer (Vertex shader)
		VkDescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		layoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo descriptorLayoutCI{};
		descriptorLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorLayoutCI.pNext = nullptr;
		descriptorLayoutCI.bindingCount = 1;
		descriptorLayoutCI.pBindings = &layoutBinding;
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutCI, nullptr, &descriptorSetLayout));
	}

	// Shaders access data using descriptor sets that "point" at our uniform buffers
	// The descriptor sets make use of the descriptor set layouts created above 
	void createDescriptorSets()
	{
		// Allocate one descriptor set per frame from the global descriptor pool
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.descriptorSetCount = 1;
			allocInfo.pSetLayouts = &descriptorSetLayout;
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &uniformBuffers[i].descriptorSet));

			// Update the descriptor set determining the shader binding points
			// For every binding point used in a shader there needs to be one
			// descriptor set matching that binding point
			VkWriteDescriptorSet writeDescriptorSet{};

			// The buffer's information is passed using a descriptor info structure
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = uniformBuffers[i].buffer;
			bufferInfo.range = sizeof(ShaderData);

			// Binding 0 : Uniform buffer
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.dstSet = uniformBuffers[i].descriptorSet;
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			writeDescriptorSet.pBufferInfo = &bufferInfo;
			writeDescriptorSet.dstBinding = 0;
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
	}

	// Create the depth (and stencil) buffer attachments used by our framebuffers
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupDepthStencil()
	{
		// Create an optimal image used as the depth stencil attachment
		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = depthFormat;
		// Use example's height and width
		imageCI.extent = { width, height, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &depthStencil.image));

		// Allocate memory for the image (device local) and bind it to our image
		VkMemoryAllocateInfo memAlloc{};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &depthStencil.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.memory, 0));

		// Create a view for the depth stencil image
		// Images aren't directly accessed in Vulkan, but rather through views described by a subresource range
		// This allows for multiple views of one image with differing ranges (e.g. for different layers)
		VkImageViewCreateInfo depthStencilViewCI{};
		depthStencilViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		depthStencilViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilViewCI.format = depthFormat;
		depthStencilViewCI.subresourceRange = {};
		depthStencilViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT)
		if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
			depthStencilViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
		depthStencilViewCI.subresourceRange.baseMipLevel = 0;
		depthStencilViewCI.subresourceRange.levelCount = 1;
		depthStencilViewCI.subresourceRange.baseArrayLayer = 0;
		depthStencilViewCI.subresourceRange.layerCount = 1;
		depthStencilViewCI.image = depthStencil.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilViewCI, nullptr, &depthStencil.view));
	}

	// Create a frame buffer for each swap chain image
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupFrameBuffer()
	{
		// Create a frame buffer for every image in the swapchain
		frameBuffers.resize(swapChain.images.size());
		for (size_t i = 0; i < frameBuffers.size(); i++)
		{
			std::array<VkImageView, 2> attachments{};
			// Color attachment is the view of the swapchain image
			attachments[0] = swapChain.imageViews[i];
			// Depth/Stencil attachment is the same for all frame buffers due to how depth works with current GPUs
			attachments[1] = depthStencil.view;

			VkFramebufferCreateInfo frameBufferCI{};
			frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			// All frame buffers use the same renderpass setup
			frameBufferCI.renderPass = renderPass;
			frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCI.pAttachments = attachments.data();
			frameBufferCI.width = width;
			frameBufferCI.height = height;
			frameBufferCI.layers = 1;
			// Create the framebuffer
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &frameBuffers[i]));
		}
	}

	// Render pass setup
	// Render passes are a new concept in Vulkan. They describe the attachments used during rendering and may contain multiple subpasses with attachment dependencies
	// This allows the driver to know up-front what the rendering will look like and is a good opportunity to optimize especially on tile-based renderers (with multiple subpasses)
	// Using sub pass dependencies also adds implicit layout transitions for the attachment used, so we don't need to add explicit image memory barriers to transform them
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupRenderPass()
	{
		// This example will use a single render pass with one subpass

		// Descriptors for the attachments used by this renderpass
		std::array<VkAttachmentDescription, 2> attachments{};

		// Color attachment
		attachments[0].format = swapChain.colorFormat;                                  // Use the color format selected by the swapchain
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;                                 // We don't use multi sampling in this example
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                            // Clear this attachment at the start of the render pass
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;                          // Keep its contents after the render pass is finished (for displaying it)
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                 // We don't use stencil, so don't care for load
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;               // Same for store
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                       // Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;                   // Layout to which the attachment is transitioned when the render pass is finished
		// As we want to present the color buffer to the swapchain, we transition to PRESENT_KHR
// Depth attachment
		attachments[1].format = depthFormat;                                           // A proper depth format is selected in the example base
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                           // Clear depth at start of first subpass
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;                     // We don't need depth after render pass has finished (DONT_CARE may result in better performance)
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                // No stencil
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;              // No Stencil
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                      // Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // Transition to depth/stencil attachment

		// Setup attachment references
		VkAttachmentReference colorReference{};
		colorReference.attachment = 0;                                    // Attachment 0 is color
		colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Attachment layout used as color during the subpass

		VkAttachmentReference depthReference{};
		depthReference.attachment = 1;                                            // Attachment 1 is color
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // Attachment used as depth/stencil used during the subpass

		// Setup a single subpass reference
		VkSubpassDescription subpassDescription{};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;                            // Subpass uses one color attachment
		subpassDescription.pColorAttachments = &colorReference;                 // Reference to the color attachment in slot 0
		subpassDescription.pDepthStencilAttachment = &depthReference;           // Reference to the depth attachment in slot 1
		subpassDescription.inputAttachmentCount = 0;                            // Input attachments can be used to sample from contents of a previous subpass
		subpassDescription.pInputAttachments = nullptr;                         // (Input attachments not used by this example)
		subpassDescription.preserveAttachmentCount = 0;                         // Preserved attachments can be used to loop (and preserve) attachments through subpasses
		subpassDescription.pPreserveAttachments = nullptr;                      // (Preserve attachments not used by this example)
		subpassDescription.pResolveAttachments = nullptr;                       // Resolve attachments are resolved at the end of a sub pass and can be used for e.g. multi sampling

		// Setup subpass dependencies
		// These will add the implicit attachment layout transitions specified by the attachment descriptions
		// The actual usage layout is preserved through the layout specified in the attachment reference
		// Each subpass dependency will introduce a memory and execution dependency between the source and dest subpass described by
		// srcStageMask, dstStageMask, srcAccessMask, dstAccessMask (and dependencyFlags is set)
		// Note: VK_SUBPASS_EXTERNAL is a special constant that refers to all commands executed outside of the actual renderpass)
		std::array<VkSubpassDependency, 2> dependencies{};

		// Does the transition from final to initial layout for the depth an color attachments
		// Depth attachment
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		dependencies[0].dependencyFlags = 0;
		// Color attachment
		dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].dstSubpass = 0;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].srcAccessMask = 0;
		dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
		dependencies[1].dependencyFlags = 0;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassCI{};
		renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCI.attachmentCount = static_cast<uint32_t>(attachments.size());  // Number of attachments used by this render pass
		renderPassCI.pAttachments = attachments.data();                            // Descriptions of the attachments used by the render pass
		renderPassCI.subpassCount = 1;                                             // We only use one subpass in this example
		renderPassCI.pSubpasses = &subpassDescription;                             // Description of that subpass
		renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size()); // Number of subpass dependencies
		renderPassCI.pDependencies = dependencies.data();                          // Subpass dependencies used by the render pass
		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderPass));
	}

	// Vulkan loads its shaders from an immediate binary representation called SPIR-V
	// Shaders are compiled offline from e.g. GLSL using the reference glslang compiler
	// This function loads such a shader from a binary file and returns a shader module structure
	VkShaderModule loadSPIRVShader(std::string filename)
	{
		size_t shaderSize;
		char* shaderCode{ nullptr };

#if defined(__ANDROID__)
		// Load shader from compressed asset
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		shaderSize = AAsset_getLength(asset);
		assert(shaderSize > 0);

		shaderCode = new char[shaderSize];
		AAsset_read(asset, shaderCode, shaderSize);
		AAsset_close(asset);
#else
		std::ifstream is(filename, std::ios::binary | std::ios::in | std::ios::ate);

		if (is.is_open())
		{
			shaderSize = is.tellg();
			is.seekg(0, std::ios::beg);
			// Copy file contents into a buffer
			shaderCode = new char[shaderSize];
			is.read(shaderCode, shaderSize);
			is.close();
			assert(shaderSize > 0);
		}
#endif
		if (shaderCode)
		{
			// Create a new shader module that will be used for pipeline creation
			VkShaderModuleCreateInfo shaderModuleCI{};
			shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			shaderModuleCI.codeSize = shaderSize;
			shaderModuleCI.pCode = (uint32_t*)shaderCode;

			VkShaderModule shaderModule;
			VK_CHECK_RESULT(vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule));

			delete[] shaderCode;

			return shaderModule;
		}
		else
		{
			std::cerr << "Error: Could not open shader file \"" << filename << "\"" << std::endl;
			return VK_NULL_HANDLE;
		}
	}

	void verifyShaderModule(VkShaderModule module, const char* name) {
		if (module == VK_NULL_HANDLE) {
			DEBUG_LOG("ERROR: Shader module %s is null!", name);
			return;
		}

		DEBUG_LOG("Shader module %s created successfully: %p", name, (void*)module);

		// Get shader module info
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

		DEBUG_LOG("Shader module properties:");
		DEBUG_LOG("  Handle: %p", (void*)module);
	}

	void createPipelines()
	{
		// Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
		// In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
		VkPipelineLayoutCreateInfo pipelineLayoutCI{};
		pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCI.pNext = nullptr;
		pipelineLayoutCI.setLayoutCount = 1;
		pipelineLayoutCI.pSetLayouts = &descriptorSetLayout;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

		// Create the graphics pipeline used in this example
		// Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
		// A pipeline is then stored and hashed on the GPU making pipeline changes very fast
		// Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)

		VkGraphicsPipelineCreateInfo pipelineCI{};
		pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		// The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
		pipelineCI.layout = pipelineLayout;
		// Renderpass this pipeline is attached to
		pipelineCI.renderPass = renderPass;

		// Construct the different states making up the pipeline

		// Input assembly state describes how primitives are assembled
		// This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
		inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
		rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
		rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationStateCI.depthClampEnable = VK_FALSE;
		rasterizationStateCI.rasterizerDiscardEnable = VK_FALSE;
		rasterizationStateCI.depthBiasEnable = VK_FALSE;
		rasterizationStateCI.lineWidth = 1.0f;

		// Color blend state describes how blend factors are calculated (if used)
		// We need one blend attachment state per color attachment (even if blending is not used)
		VkPipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.colorWriteMask = 0xf;
		blendAttachmentState.blendEnable = VK_FALSE;
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
		colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlendStateCI.attachmentCount = 1;
		colorBlendStateCI.pAttachments = &blendAttachmentState;

		// Viewport state sets the number of viewports and scissor used in this pipeline
		// Note: This is actually overridden by the dynamic states (see below)
		VkPipelineViewportStateCreateInfo viewportStateCI{};
		viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportStateCI.viewportCount = 1;
		viewportStateCI.scissorCount = 1;

		// Enable dynamic states
		// Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
		// To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
		// For this example we will set the viewport and scissor using dynamic states
		std::vector<VkDynamicState> dynamicStateEnables;
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
		VkPipelineDynamicStateCreateInfo dynamicStateCI{};
		dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
		dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

		// Depth and stencil state containing depth and stencil compare and test operations
		// We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
		depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencilStateCI.depthTestEnable = VK_TRUE;
		depthStencilStateCI.depthWriteEnable = VK_TRUE;
		depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
		depthStencilStateCI.back.failOp = VK_STENCIL_OP_KEEP;
		depthStencilStateCI.back.passOp = VK_STENCIL_OP_KEEP;
		depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;
		depthStencilStateCI.stencilTestEnable = VK_FALSE;
		depthStencilStateCI.front = depthStencilStateCI.back;

		// Multi sampling state
		// This example does not make use of multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
		VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
		multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleStateCI.pSampleMask = nullptr;

		// Vertex input descriptions
		// Specifies the vertex input parameters for a pipeline

		// Vertex input binding
		// This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
		VkVertexInputBindingDescription vertexInputBinding{};
		vertexInputBinding.binding = 0;
		vertexInputBinding.stride = sizeof(Vertex);
		vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// Input attribute bindings describe shader attribute locations and memory layouts
		std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributs{};
		// These match the following shader layout (see triangle.vert):
		//	layout (location = 0) in vec3 inPos;
		//	layout (location = 1) in vec3 inColor;
		// Attribute location 0: Position
		vertexInputAttributs[0].binding = 0;
		vertexInputAttributs[0].location = 0;
		// Position attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
		vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributs[0].offset = offsetof(Vertex, position);
		// Attribute location 1: Color
		vertexInputAttributs[1].binding = 0;
		vertexInputAttributs[1].location = 1;
		// Color attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
		vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributs[1].offset = offsetof(Vertex, color);

		// Vertex input state used for pipeline creation
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
		vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputStateCI.vertexBindingDescriptionCount = 1;
		vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputStateCI.vertexAttributeDescriptionCount = 2;
		vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributs.data();

		// Shaders
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		// Vertex shader
		shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		// Load binary SPIR-V shader
		shaderStages[0].module = loadSPIRVShader(getShadersPath() + "triangle/triangle.vert.spv");
		// Main entry point for the shader
		shaderStages[0].pName = "main";
		assert(shaderStages[0].module != VK_NULL_HANDLE);

		// Fragment shader
		shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		// Load binary SPIR-V shader
		shaderStages[1].module = loadSPIRVShader(getShadersPath() + "triangle/triangle.frag.spv");
		// Main entry point for the shader
		shaderStages[1].pName = "main";
		assert(shaderStages[1].module != VK_NULL_HANDLE);


		verifyShaderModule(shaderStages[0].module, "Vertex");
		verifyShaderModule(shaderStages[1].module, "Fragment");
		// Set pipeline shader stage info
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		// Assign the pipeline states to the pipeline creation info structure
		pipelineCI.pVertexInputState = &vertexInputStateCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;

		// Create rendering pipeline using the specified states
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

		// Shader modules are no longer needed once the graphics pipeline has been created
		vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
		vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
	}

	void createUniformBuffers() {
		// Size of the uniform buffer object
		VkDeviceSize bufferSize = sizeof(ShaderData);

		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			// Create buffer
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = bufferSize;
			bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffers[i].buffer));

			// Get memory requirements
			VkMemoryRequirements memReqs;
			vkGetBufferMemoryRequirements(device, uniformBuffers[i].buffer, &memReqs);

			// Allocate memory
			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memReqs.size;
			allocInfo.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &uniformBuffers[i].memory));

			// Bind buffer and memory
			VK_CHECK_RESULT(vkBindBufferMemory(device, uniformBuffers[i].buffer, uniformBuffers[i].memory, 0));

			// Map memory
			void* data;
			VK_CHECK_RESULT(vkMapMemory(device, uniformBuffers[i].memory, 0, bufferSize, 0, &data));
			uniformBuffers[i].mapped = static_cast<uint8_t*>(data);

			// Optional: Initialize the memory (if needed)
			ShaderData initialData{};
			memcpy(uniformBuffers[i].mapped, &initialData, sizeof(ShaderData));
		}
	}

	void recordPassthroughRenderCommands(VkCommandBuffer cmd, RenderTarget& sourceRT, VkFramebuffer targetFB) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &beginInfo));

		// Transition source image to shader read
		VkImageMemoryBarrier imageBarrier{};
		imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageBarrier.image = sourceRT.image;
		imageBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageBarrier);

		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 1.0f } };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = targetFB;
		renderPassInfo.renderArea.extent = { sourceRT.width, sourceRT.height };
		renderPassInfo.clearValueCount = 2;
		renderPassInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Set viewport and scissor
		VkViewport viewport{};
		viewport.width = static_cast<float>(sourceRT.width);
		viewport.height = static_cast<float>(sourceRT.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.extent = { sourceRT.width, sourceRT.height };
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		// Bind pipeline and descriptor set
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, passthrough.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			passthrough.pipelineLayout, 0, 1, &passthrough.descriptorSets[currentFrame],
			0, nullptr);

		// Draw full-screen quad (generates vertices in vertex shader)
		vkCmdDraw(cmd, 3, 1, 0, 0);

		vkCmdEndRenderPass(cmd);

		// Transition source image back to color attachment
		imageBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		imageBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageBarrier);

		VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
	}
	void presentFrame(uint32_t imageIndex) {
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &finalCopyStage.completeSemaphores[currentFrame];
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapChain.swapChain;
		presentInfo.pImageIndices = &imageIndex;
		VK_CHECK_RESULT(vkQueuePresentKHR(queue, &presentInfo));
	}


	void submitStages(uint32_t imageIndex) {
		// For first render
		VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		// For subsequent stages, wait on fragment shader
		VkPipelineStageFlags shaderWaitStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		// Submit initial render
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &presentCompleteSemaphores[currentFrame];
		submitInfo.pWaitDstStageMask = &waitStage;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &renderStage.commandBuffers[currentFrame];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderStage.completeSemaphores[currentFrame];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, renderStage.fences[currentFrame]));

		// RT1 to RT2
		submitInfo.pWaitSemaphores = &renderStage.completeSemaphores[currentFrame];
		submitInfo.pWaitDstStageMask = &shaderWaitStage;
		submitInfo.pCommandBuffers = &firstCopyStage.commandBuffers[currentFrame];
		submitInfo.pSignalSemaphores = &firstCopyStage.completeSemaphores[currentFrame];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, firstCopyStage.fences[currentFrame]));

		// RT2 to RT3
		submitInfo.pWaitSemaphores = &firstCopyStage.completeSemaphores[currentFrame];
		submitInfo.pCommandBuffers = &secondCopyStage.commandBuffers[currentFrame];
		submitInfo.pSignalSemaphores = &secondCopyStage.completeSemaphores[currentFrame];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, secondCopyStage.fences[currentFrame]));

		// RT3 to Swapchain
		submitInfo.pWaitSemaphores = &secondCopyStage.completeSemaphores[currentFrame];
		submitInfo.pCommandBuffers = &finalCopyStage.commandBuffers[currentFrame];
		submitInfo.pSignalSemaphores = &finalCopyStage.completeSemaphores[currentFrame];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, finalCopyStage.fences[currentFrame]));
	}

	void updatePassthroughDescriptorSet(PassthroughStage& stage, RenderTarget& sourceRT, uint32_t frameIndex) {
		// First verify we have valid views before proceeding
		if (sourceRT.views.empty()) {
			throw std::runtime_error("Source render target has no views!");
		}

		if (stage.isYuvInput && sourceRT.views.size() < 2) {
			throw std::runtime_error("YUV input requires at least 2 views!");
		}

		std::vector<VkWriteDescriptorSet> descriptorWrites;
		std::vector<VkDescriptorImageInfo> imageInfos;  // Keep alive until vkUpdateDescriptorSets

		// Create the descriptor writes based on valid views
		VkDescriptorImageInfo imageInfo{};
		imageInfo.sampler = sourceRT.sampler;
		imageInfo.imageView = sourceRT.views[0];  // First view
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfos.push_back(imageInfo);

		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = stage.descriptorSets[frameIndex];
		write.dstBinding = 0;
		write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		write.descriptorCount = 1;
		write.pImageInfo = &imageInfos.back();
		descriptorWrites.push_back(write);

		// Only continue if we have valid descriptor writes
		if (!descriptorWrites.empty()) {
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),
				descriptorWrites.data(), 0, nullptr);
		}
	}

	void createPassthroughDescriptorSets(PassthroughStage& stage) {
		// Allocate descriptor sets for each frame in flight
		std::vector<VkDescriptorSetLayout> layouts(MAX_CONCURRENT_FRAMES, stage.descriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = stage.descriptorPool;
		allocInfo.descriptorSetCount = MAX_CONCURRENT_FRAMES;
		allocInfo.pSetLayouts = layouts.data();

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, stage.descriptorSets.data()));

		// Initial descriptor set update for each frame
		for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			std::vector<VkWriteDescriptorSet> descriptorWrites;

			if (stage.isYuvInput) {
				// Y plane descriptor
				VkDescriptorImageInfo yImageInfo{};
				yImageInfo.sampler = VK_NULL_HANDLE;  // Will be updated later
				yImageInfo.imageView = VK_NULL_HANDLE;
				yImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

				VkWriteDescriptorSet yWrite{};
				yWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				yWrite.dstSet = stage.descriptorSets[i];
				yWrite.dstBinding = 0;
				yWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				yWrite.descriptorCount = 1;
				yWrite.pImageInfo = &yImageInfo;
				descriptorWrites.push_back(yWrite);

				// UV plane descriptor
				VkDescriptorImageInfo uvImageInfo{};
				uvImageInfo.sampler = VK_NULL_HANDLE;  // Will be updated later
				uvImageInfo.imageView = VK_NULL_HANDLE;
				uvImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

				VkWriteDescriptorSet uvWrite{};
				uvWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				uvWrite.dstSet = stage.descriptorSets[i];
				uvWrite.dstBinding = 1;
				uvWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				uvWrite.descriptorCount = 1;
				uvWrite.pImageInfo = &uvImageInfo;
				descriptorWrites.push_back(uvWrite);
			}
			else {
				// Single RGB texture descriptor
				VkDescriptorImageInfo imageInfo{};
				imageInfo.sampler = VK_NULL_HANDLE;  // Will be updated later
				imageInfo.imageView = VK_NULL_HANDLE;
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

				VkWriteDescriptorSet write{};
				write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				write.dstSet = stage.descriptorSets[i];
				write.dstBinding = 0;
				write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				write.descriptorCount = 1;
				write.pImageInfo = &imageInfo;
				descriptorWrites.push_back(write);
			}

			// UBO descriptor
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = yuvUbo.buffer;
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(YuvUBO);

			VkWriteDescriptorSet uboWrite{};
			uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			uboWrite.dstSet = stage.descriptorSets[i];
			uboWrite.dstBinding = stage.isYuvInput ? 2 : 1;  // Binding 2 for YUV input, 1 for RGB input
			uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			uboWrite.descriptorCount = 1;
			uboWrite.pBufferInfo = &bufferInfo;
			descriptorWrites.push_back(uboWrite);

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),
				descriptorWrites.data(), 0, nullptr);
		}
	}

	void setupPassthroughStage(PassthroughStage& stage, const char* vertShaderPath, const char* fragShaderPath) {
		// Create descriptor set layout with appropriate bindings for RGB/YUV
		createPassthroughDescriptorSetLayout(stage);

		// Create pipeline layout
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &stage.descriptorSetLayout;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &stage.pipelineLayout));

		// Create descriptor pool
		std::vector<VkDescriptorPoolSize> poolSizes;

		// Add sampler pool size
		VkDescriptorPoolSize samplerPoolSize{};
		samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerPoolSize.descriptorCount = MAX_CONCURRENT_FRAMES * (stage.isYuvInput ? 2 : 1);  // 2 for YUV (Y+UV), 1 for RGB
		poolSizes.push_back(samplerPoolSize);

		// Add UBO pool size
		VkDescriptorPoolSize uboPoolSize{};
		uboPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboPoolSize.descriptorCount = MAX_CONCURRENT_FRAMES;  // One UBO per frame
		poolSizes.push_back(uboPoolSize);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.maxSets = MAX_CONCURRENT_FRAMES;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &stage.descriptorPool));

		// Create descriptor sets
		createPassthroughDescriptorSets(stage);

		// Create pipeline
		createPassthroughPipeline(stage, vertShaderPath, fragShaderPath);

		// Log stage setup for debugging
		const char* stageType;
		if (stage.isYuvInput && stage.isYuvOutput) {
			stageType = "YUV to YUV";
		}
		else if (stage.isYuvInput) {
			stageType = "YUV to RGB";
		}
		else if (stage.isYuvOutput) {
			stageType = "RGB to YUV";
		}
		else {
			stageType = "RGB to RGB";
		}
		DEBUG_LOG("Setup passthrough stage: %s", stageType);
		// Verify stage setup
		if (stage.pipeline == VK_NULL_HANDLE ||
			stage.pipelineLayout == VK_NULL_HANDLE ||
			stage.descriptorSetLayout == VK_NULL_HANDLE ||
			stage.descriptorPool == VK_NULL_HANDLE) {
			throw std::runtime_error("Failed to setup passthrough stage properly");
		}

		// Verify descriptor sets
		for (const auto& descriptorSet : stage.descriptorSets) {
			if (descriptorSet == VK_NULL_HANDLE) {
				throw std::runtime_error("Failed to create descriptor set");
			}
		}
	}

	void createPassthroughPipeline(PassthroughStage& stage, const char* vertShaderPath, const char* fragShaderPath) {
		// Load shaders
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		// Vertex shader
		shaderStages[0] = {};
		shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderStages[0].module = loadSPIRVShader(vertShaderPath);
		shaderStages[0].pName = "main";
		assert(shaderStages[0].module != VK_NULL_HANDLE);

		// Fragment shader
		shaderStages[1] = {};
		shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderStages[1].module = loadSPIRVShader(fragShaderPath);
		shaderStages[1].pName = "main";
		assert(shaderStages[1].module != VK_NULL_HANDLE);

		// Pipeline layout
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &stage.descriptorSetLayout;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &stage.pipelineLayout));

		// Pipeline state
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		// Vertex input
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		pipelineInfo.pVertexInputState = &vertexInputInfo;

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		pipelineInfo.pInputAssemblyState = &inputAssembly;

		// Viewport and scissor (dynamic)
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;
		pipelineInfo.pViewportState = &viewportState;

		// Rasterization
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.lineWidth = 1.0f;
		pipelineInfo.pRasterizationState = &rasterizer;

		// Multisampling
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		pipelineInfo.pMultisampleState = &multisampling;

		// Color blend
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		pipelineInfo.pColorBlendState = &colorBlending;

		// Dynamic states
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();
		pipelineInfo.pDynamicState = &dynamicState;

		// Depth stencil (disabled for passthrough)
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		pipelineInfo.pDepthStencilState = &depthStencil;

		// Final pipeline settings
		pipelineInfo.layout = stage.pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineInfo.pStages = shaderStages.data();

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &stage.pipeline));

		// Cleanup shader modules
		vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
		vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
	}
	void updatePassthroughDescriptorSets(PassthroughStage& stage, RenderTarget& sourceRT, uint32_t frameIndex) {
		std::vector<VkWriteDescriptorSet> descriptorWrites;
		std::vector<VkDescriptorImageInfo> imageInfos;

		if (stage.isYuvInput) {
			// Y plane
			VkDescriptorImageInfo yImageInfo{};
			yImageInfo.sampler = sourceRT.sampler;
			yImageInfo.imageView = sourceRT.views[0];  // Y plane view
			yImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos.push_back(yImageInfo);

			VkWriteDescriptorSet yWrite{};
			yWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			yWrite.dstSet = stage.descriptorSets[frameIndex];
			yWrite.dstBinding = 0;
			yWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			yWrite.descriptorCount = 1;
			yWrite.pImageInfo = &imageInfos.back();
			descriptorWrites.push_back(yWrite);

			// UV plane
			VkDescriptorImageInfo uvImageInfo{};
			uvImageInfo.sampler = sourceRT.sampler;
			uvImageInfo.imageView = sourceRT.views[1];  // UV plane view
			uvImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos.push_back(uvImageInfo);

			VkWriteDescriptorSet uvWrite = yWrite;
			uvWrite.dstBinding = 1;
			uvWrite.pImageInfo = &imageInfos.back();
			descriptorWrites.push_back(uvWrite);
		}
		else {
			// Single RGB texture
			VkDescriptorImageInfo imageInfo{};
			imageInfo.sampler = sourceRT.sampler;
			imageInfo.imageView = sourceRT.views[0];
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos.push_back(imageInfo);

			VkWriteDescriptorSet write{};
			write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = stage.descriptorSets[frameIndex];
			write.dstBinding = 0;
			write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			write.descriptorCount = 1;
			write.pImageInfo = &imageInfos.back();
			descriptorWrites.push_back(write);
		}

		// UBO descriptor
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = yuvUbo.buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(YuvUBO);

		VkWriteDescriptorSet uboWrite{};
		uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		uboWrite.dstSet = stage.descriptorSets[frameIndex];
		uboWrite.dstBinding = descriptorWrites.back().dstBinding + 1;
		uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboWrite.descriptorCount = 1;
		uboWrite.pBufferInfo = &bufferInfo;
		descriptorWrites.push_back(uboWrite);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),
			descriptorWrites.data(), 0, nullptr);
	}

	void recordInitialRenderCommands(uint32_t frameIndex) {
		VkCommandBufferBeginInfo cmdBufInfo{};
		cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		VkCommandBuffer cmd = renderStage.commandBuffers[frameIndex];
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));

		VkClearValue clearValues[2]{};
		clearValues[0].color = { { 1.0f, 0.0f, 0.0f, 1.0f } };  // Red background
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.framebuffer = renderTarget1Framebuffer;
		renderPassBeginInfo.renderArea.extent = { renderTarget1.width, renderTarget1.height };
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport{};
		viewport.width = (float)renderTarget1.width;
		viewport.height = (float)renderTarget1.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.extent = { renderTarget1.width, renderTarget1.height };
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		// Bind descriptor sets
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipelineLayout, 0, 1, &uniformBuffers[frameIndex].descriptorSet, 0, nullptr);

		// Bind the rendering pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		// Bind vertex and index buffers
		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, offsets);
		vkCmdBindIndexBuffer(cmd, indices.buffer, 0, VK_INDEX_TYPE_UINT32);

		// Draw indexed triangle
		vkCmdDrawIndexed(cmd, indices.count, 1, 0, 0, 0);

		vkCmdEndRenderPass(cmd);

		VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
	}
	void dumpImage(VkImage image, uint32_t width, uint32_t height, const char* filename) {
		// Create a buffer to store the image data
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = width * height * 4;  // Assuming RGBA8
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

		VkBuffer stagingBuffer;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer));

		// Allocate memory for the buffer
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VkDeviceMemory stagingMemory;
		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));

		// Copy image to buffer
		VkCommandBuffer cmdBuffer;
		VkCommandBufferAllocateInfo cmdBufAllocInfo{};
		cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocInfo.commandPool = commandPool;
		cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocInfo.commandBufferCount = 1;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &cmdBuffer));

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		VkBufferImageCopy region{};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.layerCount = 1;
		region.imageExtent = { width, height, 1 };

		vkCmdCopyImageToBuffer(cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			stagingBuffer, 1, &region);

		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuffer;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));

		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		// Map memory and save to file
		void* data;
		vkMapMemory(device, stagingMemory, 0, width * height * 4, 0, &data);

		// Save to file
		std::ofstream file(filename, std::ios::binary);
		file.write(reinterpret_cast<char*>(data), width * height * 4);
		file.close();

		// Cleanup
		vkUnmapMemory(device, stagingMemory);
		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, commandPool, 1, &cmdBuffer);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingMemory, nullptr);
	}
	void dumpYUVImage(VkImage image, uint32_t width, uint32_t height, const char* filename) {
		// For NV12, we need width*height for Y plane and width*height/2 for UV plane
		VkDeviceSize totalSize = (width * height) + (width * height / 2);  // NV12 format size

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = totalSize;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

		VkBuffer stagingBuffer;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer));

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VkDeviceMemory stagingMemory;
		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));

		// Command buffer setup
		VkCommandBuffer cmdBuffer;
		VkCommandBufferAllocateInfo cmdBufAllocInfo{};
		cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocInfo.commandPool = commandPool;
		cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocInfo.commandBufferCount = 1;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &cmdBuffer));

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

		// Transition both planes to transfer src
		std::array<VkImageMemoryBarrier, 2> barriers{};

		// Y plane barrier
		barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].image = image;
		barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
		barriers[0].subresourceRange.baseMipLevel = 0;
		barriers[0].subresourceRange.levelCount = 1;
		barriers[0].subresourceRange.baseArrayLayer = 0;
		barriers[0].subresourceRange.layerCount = 1;
		barriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		// UV plane barrier
		barriers[1] = barriers[0];
		barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			barriers.size(), barriers.data());

		// Copy both planes
		std::array<VkBufferImageCopy, 2> regions{};

		// Y plane copy
		regions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
		regions[0].imageSubresource.layerCount = 1;
		regions[0].imageExtent = { width, height, 1 };
		regions[0].bufferOffset = 0;

		// UV plane copy
		regions[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
		regions[1].imageSubresource.layerCount = 1;
		regions[1].imageExtent = { width / 2, height / 2, 1 };
		regions[1].bufferOffset = width * height;  // UV data starts after Y data

		for (const auto& region : regions) {
			vkCmdCopyImageToBuffer(cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				stagingBuffer, 1, &region);
		}

		// Transition back to shader read
		barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			barriers.size(), barriers.data());

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));

		// Submit and wait
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuffer;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));

		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		// Map and save
		void* data;
		vkMapMemory(device, stagingMemory, 0, totalSize, 0, &data);

		std::ofstream file(filename, std::ios::binary);
		file.write(reinterpret_cast<char*>(data), totalSize);
		file.close();

		// Cleanup
		vkUnmapMemory(device, stagingMemory);
		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, commandPool, 1, &cmdBuffer);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingMemory, nullptr);
	}

	void verifyYUVConversion(VkImage image, uint32_t width, uint32_t height, const char* filename) {
		VkDeviceSize ySize = width * height;
		VkDeviceSize uvSize = (width * height) / 2;  // UV planes are quarter size each
		VkDeviceSize totalSize = ySize + uvSize;

		// Create staging buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = totalSize;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer));

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));

		// Copy command buffer
		VkCommandBuffer cmdBuffer;
		VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandPool = commandPool;
		cmdBufAllocateInfo.commandBufferCount = 1;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &cmdBuffer));

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

		// Transition image layout for copy
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.layerCount = 1;

		// Y plane transition
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		// UV plane transition
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		// Copy both planes
		std::array<VkBufferImageCopy, 2> regions{};

		// Y plane copy
		regions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
		regions[0].imageSubresource.mipLevel = 0;
		regions[0].imageSubresource.baseArrayLayer = 0;
		regions[0].imageSubresource.layerCount = 1;
		regions[0].imageExtent = { width, height, 1 };

		// UV plane copy
		regions[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
		regions[1].imageSubresource.mipLevel = 0;
		regions[1].imageSubresource.baseArrayLayer = 0;
		regions[1].imageSubresource.layerCount = 1;
		regions[1].imageExtent = { width / 2, height / 2, 1 };
		regions[1].bufferOffset = ySize;

		vkCmdCopyImageToBuffer(cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			stagingBuffer, static_cast<uint32_t>(regions.size()), regions.data());

		// Transition back
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		// Y plane
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		// UV plane
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
		vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));

		// Submit command buffer
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuffer;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));

		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		// Map and analyze data
		void* data;
		vkMapMemory(device, stagingMemory, 0, totalSize, 0, &data);

		uint8_t* yData = static_cast<uint8_t*>(data);
		uint8_t* uvData = yData + ySize;

		// Sample and log some values
		DEBUG_LOG("YUV Data Analysis for %s:", filename);
		DEBUG_LOG("Y Plane (first 16 bytes):");
		for (int i = 0; i < 16; i++) {
			DEBUG_LOG("Y[%d] = %d", i, yData[i]);
		}
		DEBUG_LOG("UV Plane (first 16 bytes):");
		for (int i = 0; i < 16; i++) {
			DEBUG_LOG("UV[%d] = %d", i, uvData[i]);
		}

		// Check for all zeros
		bool allYZero = true;
		bool allUVZero = true;
		for (VkDeviceSize i = 0; i < ySize && allYZero; i++) {
			if (yData[i] != 0) allYZero = false;
		}
		for (VkDeviceSize i = 0; i < uvSize && allUVZero; i++) {
			if (uvData[i] != 0) allUVZero = false;
		}

		DEBUG_LOG("Analysis Results:");
		DEBUG_LOG("Y Plane all zeros: %s", allYZero ? "true" : "false");
		DEBUG_LOG("UV Plane all zeros: %s", allUVZero ? "true" : "false");

		// Cleanup
		vkUnmapMemory(device, stagingMemory);
		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, commandPool, 1, &cmdBuffer);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingMemory, nullptr);
	}
	void prepare() {
		VulkanExampleBase::prepare();
		createCommandPool();
		createCommandBuffers();
		createSynchronizationPrimitives();
		createYuvUbo();  // Create UBO for YUV conversions
		createUniformBuffers();

		// Create render targets
		createRenderTarget(renderTarget1, renderTarget1Framebuffer, false);  // RGB
		createRenderTarget(renderTarget2, renderTarget2Framebuffer, true);   // YUV
		createRenderTarget(renderTarget3, renderTarget3Framebuffer, true);   // YUV

		// Original pipeline setup for first render
		createVertexBuffer();
		createUniformBuffers();
		createDescriptorSetLayout();
		createDescriptorPool();
		createDescriptorSets();
		createPipelines();

		// Setup stages
		rt1ToRt2Stage.isYuvInput = false;
		rt1ToRt2Stage.isYuvOutput = true;
		setupPassthroughStage(rt1ToRt2Stage,
			"../shaders/glsl/triangle/passthrough.vert.spv",
			"../shaders/glsl/triangle/rgb_to_nv12.frag.spv");

		rt2ToRt3Stage.isYuvInput = true;
		rt2ToRt3Stage.isYuvOutput = true;
		setupPassthroughStage(rt2ToRt3Stage,
			"../shaders/glsl/triangle/passthrough.vert.spv",
			"../shaders/glsl/triangle/nv12_to_nv12.frag.spv");

		rt3ToSwapchainStage.isYuvInput = true;
		rt3ToSwapchainStage.isYuvOutput = false;
		setupPassthroughStage(rt3ToSwapchainStage,
			"../shaders/glsl/triangle/passthrough.vert.spv",
			"../shaders/glsl/triangle/nv12_to_rgb.frag.spv");

		prepared = true;

		verifyRenderTargetState(renderTarget1, "RT1 (RGB)");
		verifyRenderTargetState(renderTarget2, "RT2 (YUV)");
		verifyRenderTargetState(renderTarget3, "RT3 (YUV)");
	}

	void updateUniformBuffers(uint32_t currentImage) {
		// Update matrices
		ShaderData shaderData{};
		shaderData.projectionMatrix = camera.matrices.perspective;
		shaderData.viewMatrix = camera.matrices.view;
		shaderData.modelMatrix = glm::mat4(1.0f);  // Identity matrix for simple rotation

		// Map uniform buffer and update it
		memcpy(uniformBuffers[currentImage].mapped, &shaderData, sizeof(ShaderData));
	}

	void verifyRenderTargetState(RenderTarget& rt, const char* debugName) {
		DEBUG_LOG("Verifying render target: %s", debugName);
		DEBUG_LOG("Image handle: %p", (void*)rt.image);
		DEBUG_LOG("Format: %d", rt.format);
		DEBUG_LOG("Number of views: %zu", rt.views.size());
		DEBUG_LOG("Is YUV: %s", rt.isYuv ? "true" : "false");

		// Verify memory allocation
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, rt.image, &memReqs);
		DEBUG_LOG("Memory size: %zu", memReqs.size);
		DEBUG_LOG("Memory alignment: %zu", memReqs.alignment);

		// Additional YUV-specific checks
		if (rt.isYuv) {
			VkImageSubresource subRes{};
			subRes.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
			VkSubresourceLayout layout;
			vkGetImageSubresourceLayout(device, rt.image, &subRes, &layout);
			DEBUG_LOG("Y plane offset: %zu", layout.offset);
			DEBUG_LOG("Y plane row pitch: %zu", layout.rowPitch);

			subRes.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
			vkGetImageSubresourceLayout(device, rt.image, &subRes, &layout);
			DEBUG_LOG("UV plane offset: %zu", layout.offset);
			DEBUG_LOG("UV plane row pitch: %zu", layout.rowPitch);
		}
	}

	void debugDescriptorSetState(PassthroughStage& stage, uint32_t frameIndex) {
		DEBUG_LOG("Verifying descriptor set state for frame %d:", frameIndex);

		// Check descriptor set handle
		DEBUG_LOG("Descriptor set handle: %p", (void*)stage.descriptorSets[frameIndex]);

		// Verify descriptor set layout
		DEBUG_LOG("Descriptor set layout handle: %p", (void*)stage.descriptorSetLayout);

		// Print pipeline layout
		DEBUG_LOG("Pipeline layout handle: %p", (void*)stage.pipelineLayout);

		// Print pipeline handle
		DEBUG_LOG("Pipeline handle: %p", (void*)stage.pipeline);
	}

	void debugRenderPassState(VkCommandBuffer cmd, PassthroughStage& stage,
		RenderTarget& sourceRT, VkFramebuffer targetFB,
		uint32_t frameIndex) {
		DEBUG_LOG("\nRender Pass State Debug:");
		DEBUG_LOG("Command Buffer: %p", (void*)cmd);
		DEBUG_LOG("Source RT image: %p", (void*)sourceRT.image);
		DEBUG_LOG("Target Framebuffer: %p", (void*)targetFB);
		DEBUG_LOG("Current Frame: %d", frameIndex);

		if (sourceRT.isYuv) {
			DEBUG_LOG("Source is YUV");
			DEBUG_LOG("Y View: %p", (void*)sourceRT.views[0]);
			DEBUG_LOG("UV View: %p", (void*)sourceRT.views[1]);
		}
		else {
			DEBUG_LOG("Source is RGB");
			DEBUG_LOG("View: %p", (void*)sourceRT.views[0]);
		}

		// Print viewport info
		DEBUG_LOG("Viewport: width=%d, height=%d", sourceRT.width, sourceRT.height);
	}

	void render() {
		if (!prepared) {
			return;
		}

		// Wait for all stages to complete from previous frame
		std::array<VkFence, 4> stageFences = {
			renderStage.fences[currentFrame],
			firstCopyStage.fences[currentFrame],
			secondCopyStage.fences[currentFrame],
			finalCopyStage.fences[currentFrame]
		};

		VK_CHECK_RESULT(vkWaitForFences(device, stageFences.size(), stageFences.data(), VK_TRUE, UINT64_MAX));
		VK_CHECK_RESULT(vkResetFences(device, stageFences.size(), stageFences.data()));

		// Get next swapchain image
		uint32_t imageIndex;
		VK_CHECK_RESULT(vkAcquireNextImageKHR(device, swapChain.swapChain, UINT64_MAX,
			presentCompleteSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex));

		// Update uniform buffer
		updateUniformBuffers(currentFrame);

		// Reset command buffers
		vkResetCommandBuffer(renderStage.commandBuffers[currentFrame], 0);
		vkResetCommandBuffer(firstCopyStage.commandBuffers[currentFrame], 0);
		vkResetCommandBuffer(secondCopyStage.commandBuffers[currentFrame], 0);
		vkResetCommandBuffer(finalCopyStage.commandBuffers[currentFrame], 0);

		verifyRenderTargetState(renderTarget1, "RT1 (RGB)");
		verifyRenderTargetState(renderTarget2, "RT2 (YUV)");
		verifyRenderTargetState(renderTarget3, "RT3 (YUV)");

		// Record commands for each stage
		recordInitialRenderCommands(currentFrame);  // RGB render to RT1
		dumpImage(renderTarget1.image, renderTarget1.width, renderTarget1.height, "rt1_rgb.raw");

		DEBUG_LOG("Starting RGB to YUV conversion...");
		// RGB to YUV (RT1 to RT2)
		recordPassthroughCommands(firstCopyStage.commandBuffers[currentFrame],
			rt1ToRt2Stage, renderTarget1, renderTarget2Framebuffer, currentFrame);
		dumpYUVImage(renderTarget2.image, renderTarget2.width, renderTarget2.height, "rt2_yuv.raw");
		verifyYUVConversion(renderTarget2.image, renderTarget2.width, renderTarget2.height, "RT2");

		DEBUG_LOG("RGB to YUV conversion complete");
		// YUV to YUV (RT2 to RT3)
		recordPassthroughCommands(secondCopyStage.commandBuffers[currentFrame],
			rt2ToRt3Stage, renderTarget2, renderTarget3Framebuffer, currentFrame);
		dumpYUVImage(renderTarget3.image, renderTarget3.width, renderTarget3.height, "rt3_yuv.raw");

		// YUV to RGB (RT3 to Swapchain)
		recordPassthroughCommands(finalCopyStage.commandBuffers[currentFrame],
			rt3ToSwapchainStage, renderTarget3, frameBuffers[imageIndex], currentFrame);

		// Submit stages
		submitStages(imageIndex);

		// Present
		presentFrame(imageIndex);

		currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
	}

	void validatePipelineSetup(PassthroughStage& stage) {
		DEBUG_LOG("\n=== Validating Pipeline Setup ===");

		if (stage.descriptorSetLayout == VK_NULL_HANDLE) {
			DEBUG_LOG("ERROR: Descriptor set layout is null!");
			return;
		}

		if (stage.pipelineLayout == VK_NULL_HANDLE) {
			DEBUG_LOG("ERROR: Pipeline layout is null!");
			return;
		}

		if (stage.pipeline == VK_NULL_HANDLE) {
			DEBUG_LOG("ERROR: Pipeline is null!");
			return;
		}

		// Log the bindings we expect to have
		DEBUG_LOG("Expected descriptor bindings:");
		DEBUG_LOG("  Binding 0: Combined Image Sampler");
		DEBUG_LOG("  Binding 1: Uniform Buffer");

		// Log the actual pipeline layout info
		DEBUG_LOG("Pipeline Layout Information:");
		DEBUG_LOG("  Pipeline Layout Handle: %p", (void*)stage.pipelineLayout);
		DEBUG_LOG("  Descriptor Set Layout Handle: %p", (void*)stage.descriptorSetLayout);

		// Verify descriptor sets
		for (size_t i = 0; i < stage.descriptorSets.size(); i++) {
			if (stage.descriptorSets[i] == VK_NULL_HANDLE) {
				DEBUG_LOG("ERROR: Descriptor set %zu is null!", i);
			}
			else {
				DEBUG_LOG("Descriptor set %zu handle: %p", i, (void*)stage.descriptorSets[i]);
			}
		}

		DEBUG_LOG("Pipeline validation complete");
	}
	bool checkRenderTargetValid(const RenderTarget& rt, const char* rtName) {
		DEBUG_LOG("\nChecking render target: %s", rtName);

		if (!rt.image) {
			DEBUG_LOG("ERROR: %s image is null!", rtName);
			return false;
		}

		if (!rt.sampler) {
			DEBUG_LOG("ERROR: %s sampler is null!", rtName);
			return false;
		}

		if (rt.views.empty()) {
			DEBUG_LOG("ERROR: %s has no views!", rtName);
			return false;
		}

		for (size_t i = 0; i < rt.views.size(); i++) {
			if (!rt.views[i]) {
				DEBUG_LOG("ERROR: %s view %zu is null!", rtName, i);
				return false;
			}
		}

		DEBUG_LOG("%s render target is valid:", rtName);
		DEBUG_LOG("  Image: %p", (void*)rt.image);
		DEBUG_LOG("  Sampler: %p", (void*)rt.sampler);
		DEBUG_LOG("  View count: %zu", rt.views.size());
		for (size_t i = 0; i < rt.views.size(); i++) {
			DEBUG_LOG("  View %zu: %p", i, (void*)rt.views[i]);
		}

		return true;
	}
	void verifyDescriptorBindings(PassthroughStage& stage, RenderTarget& sourceRT, uint32_t frameIndex) {
		DEBUG_LOG("\n=== Verifying Descriptor Bindings ===");

		if (!sourceRT.sampler) {
			DEBUG_LOG("ERROR: Source sampler is null!");
			return;
		}

		if (sourceRT.views.empty()) {
			DEBUG_LOG("ERROR: Source views are empty!");
			return;
		}

		if (!sourceRT.views[0]) {
			DEBUG_LOG("ERROR: First source view is null!");
			return;
		}

		VkDescriptorSet descSet = stage.descriptorSets[frameIndex];
		if (!descSet) {
			DEBUG_LOG("ERROR: Descriptor set is null!");
			return;
		}

		DEBUG_LOG("Descriptor Set: %p", (void*)descSet);

		// Create the descriptor info structures
		VkDescriptorImageInfo imageInfo;
		VkDescriptorBufferInfo bufferInfo;

		// Initialize image info
		imageInfo = {};  // Clear the struct
		imageInfo.sampler = sourceRT.sampler;
		imageInfo.imageView = sourceRT.views[0];
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		// Initialize buffer info
		bufferInfo = {};  // Clear the struct
		bufferInfo.buffer = yuvUbo.buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(YuvUBO);

		// Log the info we're about to use
		DEBUG_LOG("Image Info:");
		DEBUG_LOG("  Sampler: %p", (void*)imageInfo.sampler);
		DEBUG_LOG("  ImageView: %p", (void*)imageInfo.imageView);
		DEBUG_LOG("  Layout: %d", imageInfo.imageLayout);

		DEBUG_LOG("Buffer Info:");
		DEBUG_LOG("  Buffer: %p", (void*)bufferInfo.buffer);
		DEBUG_LOG("  Offset: %zu", bufferInfo.offset);
		DEBUG_LOG("  Range: %zu", bufferInfo.range);

		// Create and verify the write descriptor sets
		VkWriteDescriptorSet writeImage = {};
		writeImage.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeImage.dstSet = descSet;
		writeImage.dstBinding = 0;
		writeImage.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writeImage.descriptorCount = 1;
		writeImage.pImageInfo = &imageInfo;

		VkWriteDescriptorSet writeBuffer = {};
		writeBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeBuffer.dstSet = descSet;
		writeBuffer.dstBinding = 1;
		writeBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeBuffer.descriptorCount = 1;
		writeBuffer.pBufferInfo = &bufferInfo;

		std::array<VkWriteDescriptorSet, 2> writes = { writeImage, writeBuffer };

		DEBUG_LOG("Updating descriptors...");
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		DEBUG_LOG("Descriptor update completed");

		// Verify the update
		DEBUG_LOG("Verification complete:");
		DEBUG_LOG("  Image descriptor updated for binding 0");
		DEBUG_LOG("  Buffer descriptor updated for binding 1");
	}
	void recordPassthroughCommands(VkCommandBuffer cmd, PassthroughStage& stage,
		RenderTarget& sourceRT, VkFramebuffer targetFB, uint32_t frameIndex) {

		DEBUG_LOG("\n=== Starting Passthrough Command Recording ===");

		// Validate render target first
		if (!checkRenderTargetValid(sourceRT, "Source")) {
			DEBUG_LOG("ERROR: Invalid source render target!");
			return;
		}

		debugDescriptorSetState(stage, frameIndex);
		debugRenderPassState(cmd, stage, sourceRT, targetFB, frameIndex);
		// Continue with pipeline validation and descriptor updates
		validatePipelineSetup(stage);
		verifyDescriptorBindings(stage, sourceRT, frameIndex);




		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &beginInfo));

		// Update UBO with current height
		YuvUBO* ubo = (YuvUBO*)yuvUbo.mapped;
		ubo->height = sourceRT.height;

		// Transition source image(s) to shader read
		std::vector<VkImageMemoryBarrier> barriers;

		if (stage.isYuvInput) {
			// Y plane barrier
			VkImageMemoryBarrier yBarrier{};
			yBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			yBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			yBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			yBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			yBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			yBarrier.image = sourceRT.image;
			yBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
			yBarrier.subresourceRange.baseMipLevel = 0;
			yBarrier.subresourceRange.levelCount = 1;
			yBarrier.subresourceRange.baseArrayLayer = 0;
			yBarrier.subresourceRange.layerCount = 1;
			barriers.push_back(yBarrier);
			// Add debug logging before the first barrier (transition to shader read)
			DEBUG_LOG("Memory Barrier Configuration (To Shader Read):");
			DEBUG_LOG("Source Access Mask: 0x%x", yBarrier.srcAccessMask);
			DEBUG_LOG("Dest Access Mask: 0x%x", yBarrier.dstAccessMask);
			DEBUG_LOG("Old Layout: %d", yBarrier.oldLayout);
			DEBUG_LOG("New Layout: %d", yBarrier.newLayout);
			// UV plane barrier
			VkImageMemoryBarrier uvBarrier = yBarrier;
			uvBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
			barriers.push_back(uvBarrier);
			// Add debug logging before the first barrier (transition to shader read)
			DEBUG_LOG("Memory Barrier Configuration (To Shader Read):");
			DEBUG_LOG("Source Access Mask: 0x%x", uvBarrier.srcAccessMask);
			DEBUG_LOG("Dest Access Mask: 0x%x", uvBarrier.dstAccessMask);
			DEBUG_LOG("Old Layout: %d", uvBarrier.oldLayout);
			DEBUG_LOG("New Layout: %d", uvBarrier.newLayout);
		}
		else {
			// Single RGB image barrier
			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.image = sourceRT.image;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barriers.push_back(barrier);
		// Add debug logging before the first barrier (transition to shader read)
		DEBUG_LOG("Memory Barrier Configuration (To Shader Read):");
		DEBUG_LOG("Source Access Mask: 0x%x", barrier.srcAccessMask);
		DEBUG_LOG("Dest Access Mask: 0x%x", barrier.dstAccessMask);
		DEBUG_LOG("Old Layout: %d", barrier.oldLayout);
		DEBUG_LOG("New Layout: %d", barrier.newLayout);
		}


		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			static_cast<uint32_t>(barriers.size()), barriers.data());

		// Begin render pass
		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = targetFB;
		renderPassInfo.renderArea.extent = { sourceRT.width, sourceRT.height };

		VkClearValue clearValues[2]{};
		clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };
		renderPassInfo.clearValueCount = 2;
		renderPassInfo.pClearValues = clearValues;


		vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Set viewport and scissor
		VkViewport viewport{};
		viewport.width = static_cast<float>(sourceRT.width);
		viewport.height = static_cast<float>(sourceRT.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.extent = { sourceRT.width, sourceRT.height };
		vkCmdSetScissor(cmd, 0, 1, &scissor);
		updatePassthroughDescriptorSet(stage, sourceRT, frameIndex);

		// Add validation before descriptor update
		validatePipelineSetup(stage);
		verifyDescriptorBindings(stage, sourceRT, frameIndex);


		// Bind pipeline and descriptor set
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stage.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			stage.pipelineLayout, 0, 1, &stage.descriptorSets[frameIndex], 0, nullptr);

		// After binding pipeline and descriptor set
		DEBUG_LOG("Pipeline bound: %p", (void*)stage.pipeline);
		DEBUG_LOG("Descriptor set bound: %p", (void*)stage.descriptorSets[frameIndex]);


		// Draw fullscreen quad
		vkCmdDraw(cmd, 3, 1, 0, 0);

		vkCmdEndRenderPass(cmd);

		// Transition back to color attachment
		for (auto& barrier : barriers) {
			std::swap(barrier.srcAccessMask, barrier.dstAccessMask);
			std::swap(barrier.oldLayout, barrier.newLayout);
		}

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0, nullptr,
			0, nullptr,
			static_cast<uint32_t>(barriers.size()), barriers.data());

		VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
	}
};

// OS specific main entry points
// Most of the code base is shared for the different supported operating systems, but stuff like message handling differs

#if defined(_WIN32)
// Windows entry point
VulkanExample* vulkanExample;
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);
	}
	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}
int APIENTRY WinMain(_In_ HINSTANCE hInstance, _In_opt_  HINSTANCE hPrevInstance, _In_ LPSTR, _In_ int)
{
	for (size_t i = 0; i < __argc; i++) { VulkanExample::args.push_back(__argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow(hInstance, WndProc);
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}

#elif defined(__ANDROID__)
// Android entry point
VulkanExample* vulkanExample;
void android_main(android_app* state)
{
	vulkanExample = new VulkanExample();
	state->userData = vulkanExample;
	state->onAppCmd = VulkanExample::handleAppCommand;
	state->onInputEvent = VulkanExample::handleAppInput;
	androidApp = state;
	vulkanExample->renderLoop();
	delete(vulkanExample);
}
#elif defined(_DIRECT2DISPLAY)

// Linux entry point with direct to display wsi
// Direct to Displays (D2D) is used on embedded platforms
VulkanExample* vulkanExample;
static void handleEvent()
{
}
int main(const int argc, const char* argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
VulkanExample* vulkanExample;
static void handleEvent(const DFBWindowEvent* event)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleEvent(event);
	}
}
int main(const int argc, const char* argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
VulkanExample* vulkanExample;
int main(const int argc, const char* argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(__linux__) || defined(__FreeBSD__)

// Linux entry point
VulkanExample* vulkanExample;
#if defined(VK_USE_PLATFORM_XCB_KHR)
static void handleEvent(const xcb_generic_event_t* event)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleEvent(event);
	}
}
#else
static void handleEvent()
{
}
#endif
int main(const int argc, const char* argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif (defined(VK_USE_PLATFORM_MACOS_MVK) || defined(VK_USE_PLATFORM_METAL_EXT)) && defined(VK_EXAMPLE_XCODE_GENERATED)
VulkanExample* vulkanExample;
int main(const int argc, const char* argv[])
{
	@autoreleasepool
	{
		for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
		vulkanExample = new VulkanExample();
		vulkanExample->initVulkan();
		vulkanExample->setupWindow(nullptr);
		vulkanExample->prepare();
		vulkanExample->renderLoop();
		delete(vulkanExample);
	}
	return 0;
}
#elif defined(VK_USE_PLATFORM_SCREEN_QNX)
VULKAN_EXAMPLE_MAIN()
#endif
