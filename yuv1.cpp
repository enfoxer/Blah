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

// Add at the top of the file
#if defined(_WIN32)
#include <windows.h>
#define DEBUG_LOG(...) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), __VA_ARGS__); \
        OutputDebugStringA(buf); \
        OutputDebugStringA("\n"); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    }
#elif defined(__ANDROID__)
#include <android/log.h>
#define DEBUG_LOG(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "VulkanExample", __VA_ARGS__))
#else
#define DEBUG_LOG(...) printf(__VA_ARGS__)
#endif

// We want to keep GPU and CPU busy. To do that we may start building a new command buffer while the previous one is still being executed
// This number defines how many frames may be worked on simultaneously at once
// Increasing this number may improve performance but will also introduce additional latency
#define MAX_CONCURRENT_FRAMES 2

class VulkanExample : public VulkanExampleBase
{
private:
	struct QueueManager {
		VkQueue queue = VK_NULL_HANDLE;  // Initialize to null

		void init(VkDevice device, uint32_t queueFamilyIndex) {
			vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
		}

		VkQueue getQueue() {
			return queue;
		}
	} queueManager;  // Now it's an actual member of VulkanExample

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
		VkImageView view{ VK_NULL_HANDLE };
		VkFormat format{ VK_FORMAT_R8G8B8A8_UNORM };
		uint32_t width{ 800 };
		uint32_t height{ 600 };
	};
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

	VkSampler ycbcrSampler{ VK_NULL_HANDLE };
	VkSamplerYcbcrConversion ycbcrConversion{ VK_NULL_HANDLE };

	PFN_vkCreateSamplerYcbcrConversion vkCreateSamplerYcbcrConversion;
	PFN_vkDestroySamplerYcbcrConversion vkDestroySamplerYcbcrConversion;

	void createYCbCrSampler() {
		// Load function pointers for YCbCr conversion
		vkCreateSamplerYcbcrConversion = reinterpret_cast<PFN_vkCreateSamplerYcbcrConversion>(
			vkGetDeviceProcAddr(device, "vkCreateSamplerYcbcrConversion"));
		vkDestroySamplerYcbcrConversion = reinterpret_cast<PFN_vkDestroySamplerYcbcrConversion>(
			vkGetDeviceProcAddr(device, "vkDestroySamplerYcbcrConversion"));

		if (!vkCreateSamplerYcbcrConversion || !vkDestroySamplerYcbcrConversion) {
			throw std::runtime_error("Failed to load YCbCr conversion function pointers");
		}

		VkSamplerYcbcrConversionCreateInfo conversionInfo{};
		conversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
		conversionInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		conversionInfo.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
		conversionInfo.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
		conversionInfo.components = {
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		};
		conversionInfo.xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
		conversionInfo.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
		conversionInfo.chromaFilter = VK_FILTER_LINEAR;

		VK_CHECK_RESULT(vkCreateSamplerYcbcrConversion(device, &conversionInfo, nullptr, &ycbcrConversion));


		VkSamplerYcbcrConversionInfo conversionSamplerInfo{};
		conversionSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
		conversionSamplerInfo.conversion = ycbcrConversion;

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.pNext = &conversionSamplerInfo;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

		VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &ycbcrSampler));
	}

	void createSynchronizationPrimitives()
	{
		// Create semaphore and fence for each frame in MAX_CONCURRENT_FRAMES
		DEBUG_LOG("Creating sync objects..."); // Debug print

		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			VkSemaphoreCreateInfo semaphoreCI{};
			semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			VkFenceCreateInfo fenceCI{};
			fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Create in signaled state

			// Present semaphore
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &presentCompleteSemaphores[i]));
			DEBUG_LOG("Created present semaphore %d\n", i);

			// Render stage
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &renderStage.completeSemaphores[i]));
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &renderStage.fences[i]));
			DEBUG_LOG("Created render stage sync %d\n", i);

			// First copy stage
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &firstCopyStage.completeSemaphores[i]));
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &firstCopyStage.fences[i]));
			DEBUG_LOG("Created first copy stage sync %d\n", i);

			// Second copy stage
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &secondCopyStage.completeSemaphores[i]));
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &secondCopyStage.fences[i]));
			DEBUG_LOG("Created second copy stage sync %d\n", i);

			// Final copy stage
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &finalCopyStage.completeSemaphores[i]));
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &finalCopyStage.fences[i]));
			DEBUG_LOG("Created final copy stage sync %d\n", i);
		}
		DEBUG_LOG("Sync objects creation complete\n");
	}

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

	~VulkanExample()
	{
		// Clean up used Vulkan resources
		// Note: Inherited destructor cleans up resources stored in base class

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, vertices.buffer, nullptr);
		vkFreeMemory(device, vertices.memory, nullptr);

		vkDestroyBuffer(device, indices.buffer, nullptr);
		vkFreeMemory(device, indices.memory, nullptr);

		cleanupStageSyncObjects(renderStage);
		cleanupStageSyncObjects(firstCopyStage);
		cleanupStageSyncObjects(secondCopyStage);
		cleanupStageSyncObjects(finalCopyStage);

		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			vkDestroySemaphore(device, presentCompleteSemaphores[i], nullptr);
			vkDestroyBuffer(device, uniformBuffers[i].buffer, nullptr);
			vkFreeMemory(device, uniformBuffers[i].memory, nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		// Clean up render targets
		vkDestroyImageView(device, renderTarget1.view, nullptr);
		vkDestroyImage(device, renderTarget1.image, nullptr);
		vkFreeMemory(device, renderTarget1.memory, nullptr);
		vkDestroyFramebuffer(device, renderTarget1Framebuffer, nullptr);

		vkDestroyImageView(device, renderTarget2.view, nullptr);
		vkDestroyImage(device, renderTarget2.image, nullptr);
		vkFreeMemory(device, renderTarget2.memory, nullptr);
		vkDestroyFramebuffer(device, renderTarget2Framebuffer, nullptr);
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
		// Only create semaphores and fences here, command buffers are handled separately
		VkSemaphoreCreateInfo semaphoreCI{};
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceCI{};
		fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &stage.completeSemaphores[i]));
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &stage.fences[i]));
		}
	}

	void createSynchronizationPrimitives2()
	{
		createStageSyncObjects(renderStage, "render");
		createStageSyncObjects(firstCopyStage, "firstCopy");
		createStageSyncObjects(finalCopyStage, "finalCopy");

		// Create present semaphores (still needed for swapchain acquisition)
		VkSemaphoreCreateInfo semaphoreCI{};
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &presentCompleteSemaphores[i]));
		}
	}

	void createRenderTarget(RenderTarget& rt, VkFramebuffer& framebuffer)
	{
		// Create image
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = rt.format;
		imageInfo.extent.width = rt.width;
		imageInfo.extent.height = rt.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // Added TRANSFER_DST_BIT
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &rt.image));

		// Allocate memory
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, rt.image, &memReqs);
		VkMemoryAllocateInfo memAlloc{};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &rt.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, rt.image, rt.memory, 0));

		// Create image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = rt.format;
		viewInfo.image = rt.image;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &rt.view));

		// Create framebuffer
		std::array<VkImageView, 2> attachments = {
			rt.view,
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
		vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, indices.buffer,	1, &copyRegion);
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

	void createUniformBuffers()
	{
		// Prepare and initialize the per-frame uniform buffer blocks containing shader uniforms
		// Single uniforms like in OpenGL are no longer present in Vulkan. All hader uniforms are passed via uniform buffer blocks
		VkMemoryRequirements memReqs;

		// Vertex shader uniform buffer block
		VkBufferCreateInfo bufferInfo{};
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.pNext = nullptr;
		allocInfo.allocationSize = 0;
		allocInfo.memoryTypeIndex = 0;

		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(ShaderData);
		// This buffer will be used as a uniform buffer
		bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

		// Create the buffers
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffers[i].buffer));
			// Get memory requirements including size, alignment and memory type
			vkGetBufferMemoryRequirements(device, uniformBuffers[i].buffer, &memReqs);
			allocInfo.allocationSize = memReqs.size;
			// Get the memory type index that supports host visible memory access
			// Most implementations offer multiple memory types and selecting the correct one to allocate memory from is crucial
			// We also want the buffer to be host coherent so we don't have to flush (or sync after every update.
			// Note: This may affect performance so you might not want to do this in a real world application that updates buffers on a regular base
			allocInfo.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			// Allocate memory for the uniform buffer
			VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &(uniformBuffers[i].memory)));
			// Bind memory to buffer
			VK_CHECK_RESULT(vkBindBufferMemory(device, uniformBuffers[i].buffer, uniformBuffers[i].memory, 0));
			// We map the buffer once, so we can update it without having to map it again
			VK_CHECK_RESULT(vkMapMemory(device, uniformBuffers[i].memory, 0, sizeof(ShaderData), 0, (void**)&uniformBuffers[i].mapped));
		}

	}

	void initYcbcrExtension() {
		// Check instance extensions
		uint32_t instanceExtCount;
		VK_CHECK_RESULT(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, nullptr));
		std::vector<VkExtensionProperties> instanceExts(instanceExtCount);
		VK_CHECK_RESULT(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, instanceExts.data()));

		DEBUG_LOG("Available Instance Extensions:");
		for (const auto& ext : instanceExts) {
			DEBUG_LOG("  %s", ext.extensionName);
		}

		// Check physical device extensions
		uint32_t deviceExtCount;
		VK_CHECK_RESULT(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtCount, nullptr));
		std::vector<VkExtensionProperties> deviceExts(deviceExtCount);
		VK_CHECK_RESULT(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtCount, deviceExts.data()));

		DEBUG_LOG("\nAvailable Device Extensions:");
		for (const auto& ext : deviceExts) {
			DEBUG_LOG("  %s", ext.extensionName);
		}

		// Check features
		VkPhysicalDeviceFeatures2 features2{};
		features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

		VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcrFeatures{};
		ycbcrFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
		ycbcrFeatures.pNext = nullptr;

		features2.pNext = &ycbcrFeatures;

		vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

		DEBUG_LOG("\nYCbCr Conversion Feature Support:");
		DEBUG_LOG("  samplerYcbcrConversion: %d", ycbcrFeatures.samplerYcbcrConversion);

		// List available function pointers
		DEBUG_LOG("\nFunction pointer availability:");
		auto ptr1 = vkGetDeviceProcAddr(device, "vkCreateSamplerYcbcrConversion");
		auto ptr2 = vkGetDeviceProcAddr(device, "vkCreateSamplerYcbcrConversionKHR");
		DEBUG_LOG("  vkCreateSamplerYcbcrConversion: %p", ptr1);
		DEBUG_LOG("  vkCreateSamplerYcbcrConversionKHR: %p", ptr2);
	}

	void getEnabledExtensions() 
	{
		enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

		// Add device extension
		enabledDeviceExtensions.push_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);

		// Enable required device extensions
		std::vector<const char*> deviceExtensions = {
			VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
			VK_KHR_MAINTENANCE1_EXTENSION_NAME,
			VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME
		};

		// Setup device creation info
		VkDeviceCreateInfo deviceCreateInfo{};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		// Enable YCbCr features
		VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcrFeatures{};
		ycbcrFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
		ycbcrFeatures.samplerYcbcrConversion = VK_TRUE;
		ycbcrFeatures.pNext = nullptr;

		// Link to device features 2
		VkPhysicalDeviceFeatures2 deviceFeatures2{};
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		deviceFeatures2.pNext = &ycbcrFeatures;

		// Set up device create info
		deviceCreateInfo.pNext = &deviceFeatures2;
		deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

	}

	void prepare()
	{

		// Add instance extension
		enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

		// Add device extension
		enabledDeviceExtensions.push_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);


		VkPhysicalDeviceFeatures2 deviceFeatures2{};
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

		VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcrFeatures{};
		ycbcrFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
		ycbcrFeatures.samplerYcbcrConversion = VK_TRUE;  // Request this feature
		ycbcrFeatures.pNext = nullptr;

		deviceFeatures2.pNext = &ycbcrFeatures;

		// When creating device, pass deviceFeatures2 via pNext
		VkDeviceCreateInfo deviceCreateInfo{};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.pNext = &deviceFeatures2;



		VulkanExampleBase::prepare();
		initYcbcrExtension();  // Add this line
		createCommandPool();
		createCommandBuffers();  // Allocate all stage command buffers
		createSynchronizationPrimitives();  // Create semaphores and fences
		createYCbCrSampler(); //yuv
		createRenderTargets();
		createVertexBuffer();
		createUniformBuffers();
		createDescriptorSetLayout();
		createDescriptorPool();
		createDescriptorSets();
		createPipelines();
		queueManager.init(device, swapChain.queueNodeIndex);

		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;

		// Wait for all stages to complete from previous frame
		std::array<VkFence, 3> stageFences = {
			renderStage.fences[currentFrame],
			firstCopyStage.fences[currentFrame],
			finalCopyStage.fences[currentFrame]
		};
		vkWaitForFences(device, stageFences.size(), stageFences.data(), VK_TRUE, UINT64_MAX);
		vkResetFences(device, stageFences.size(), stageFences.data());

		// Get next swapchain image
		uint32_t imageIndex;
		VK_CHECK_RESULT(vkAcquireNextImageKHR(device, swapChain.swapChain, UINT64_MAX, presentCompleteSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex));

		// Update uniform buffer
		ShaderData shaderData{};
		shaderData.projectionMatrix = camera.matrices.perspective;
		shaderData.viewMatrix = camera.matrices.view;
		shaderData.modelMatrix = glm::mat4(1.0f);
		memcpy(uniformBuffers[currentFrame].mapped, &shaderData, sizeof(ShaderData));

		// Reset command buffers
		vkResetCommandBuffer(renderStage.commandBuffers[currentFrame], 0);
		vkResetCommandBuffer(firstCopyStage.commandBuffers[currentFrame], 0);
		vkResetCommandBuffer(finalCopyStage.commandBuffers[currentFrame], 0);

		// Stage 1: Render to RT1
		{
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			VkCommandBuffer cmd = renderStage.commandBuffers[currentFrame];

			VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));

			VkClearValue clearValues[2]{};
			clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 1.0f } };
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

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &uniformBuffers[currentFrame].descriptorSet, 0, nullptr);
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, offsets);
			vkCmdBindIndexBuffer(cmd, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, indices.count, 1, 0, 0, 0);

			vkCmdEndRenderPass(cmd);

			VK_CHECK_RESULT(vkEndCommandBuffer(cmd));

			// For first stage (render)
			VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
			// First stage (render)
			VkSubmitInfo renderSubmitInfo{};
			renderSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			renderSubmitInfo.commandBufferCount = 1;
			renderSubmitInfo.pCommandBuffers = &renderStage.commandBuffers[currentFrame];
			renderSubmitInfo.waitSemaphoreCount = 1;
			renderSubmitInfo.pWaitSemaphores = &presentCompleteSemaphores[currentFrame];
			renderSubmitInfo.pWaitDstStageMask = waitStages;
			renderSubmitInfo.signalSemaphoreCount = 1;
			renderSubmitInfo.pSignalSemaphores = &renderStage.completeSemaphores[currentFrame];
			VK_CHECK_RESULT(vkQueueSubmit(queueManager.getQueue(), 1, &renderSubmitInfo, renderStage.fences[currentFrame]));

		}

		// Stage 2: Copy RT1 to RT2
		{
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			VkCommandBuffer cmd = firstCopyStage.commandBuffers[currentFrame];

			VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));

			// Transition RT1 to transfer source
			VkImageMemoryBarrier rt1TransferBarrier{};
			rt1TransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			rt1TransferBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			rt1TransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			rt1TransferBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			rt1TransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			rt1TransferBarrier.image = renderTarget1.image;
			rt1TransferBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			// Transition RT2 to transfer destination
			VkImageMemoryBarrier rt2TransferBarrier{};
			rt2TransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			rt2TransferBarrier.srcAccessMask = 0;
			rt2TransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			rt2TransferBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			rt2TransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			rt2TransferBarrier.image = renderTarget2.image;
			rt2TransferBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			VkImageMemoryBarrier barriers[] = { rt1TransferBarrier, rt2TransferBarrier };
			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				2, barriers);

			// Copy RT1 to RT2
			VkImageCopy copyRegion{};
			copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.extent = { renderTarget1.width, renderTarget1.height, 1 };

			vkCmdCopyImage(cmd,
				renderTarget1.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				renderTarget2.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &copyRegion);

			// Transition RT1 back to color attachment
			rt1TransferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			rt1TransferBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			rt1TransferBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			rt1TransferBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &rt1TransferBarrier);

			VK_CHECK_RESULT(vkEndCommandBuffer(cmd));

			// Submit first copy commands

			// For second stage (first copy)
			VkPipelineStageFlags copyWaitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
			submitInfo.pWaitDstStageMask = copyWaitStages;
			// Second stage (first copy)
			VkSubmitInfo firstCopySubmitInfo{};
			firstCopySubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			firstCopySubmitInfo.commandBufferCount = 1;
			firstCopySubmitInfo.pCommandBuffers = &firstCopyStage.commandBuffers[currentFrame];
			firstCopySubmitInfo.waitSemaphoreCount = 1;
			firstCopySubmitInfo.pWaitSemaphores = &renderStage.completeSemaphores[currentFrame];
			firstCopySubmitInfo.pWaitDstStageMask = copyWaitStages;
			firstCopySubmitInfo.signalSemaphoreCount = 1;
			firstCopySubmitInfo.pSignalSemaphores = &firstCopyStage.completeSemaphores[currentFrame];
			VK_CHECK_RESULT(vkQueueSubmit(queueManager.getQueue(), 1, &firstCopySubmitInfo, firstCopyStage.fences[currentFrame]));

		}

		// Stage 3: Copy RT2 to RT3
		{
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			VkCommandBuffer cmd = secondCopyStage.commandBuffers[currentFrame];

			VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));

			// Transition RT2 to transfer source
			VkImageMemoryBarrier rt2TransferBarrier{};
			rt2TransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			rt2TransferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			rt2TransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			rt2TransferBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			rt2TransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			rt2TransferBarrier.image = renderTarget2.image;
			rt2TransferBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			// Transition RT3 to transfer destination
			VkImageMemoryBarrier rt3TransferBarrier{};
			rt3TransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			rt3TransferBarrier.srcAccessMask = 0;
			rt3TransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			rt3TransferBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			rt3TransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			rt3TransferBarrier.image = renderTarget3.image;
			rt3TransferBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			VkImageMemoryBarrier barriers[] = { rt2TransferBarrier, rt3TransferBarrier };
			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				2, barriers);

			// Copy RT2 to RT3
			VkImageCopy copyRegion{};
			copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.extent = { renderTarget2.width, renderTarget2.height, 1 };

			vkCmdCopyImage(cmd,
				renderTarget2.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				renderTarget3.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &copyRegion);

			VK_CHECK_RESULT(vkEndCommandBuffer(cmd));

			DEBUG_LOG("About to submit second copy stage\n");
			if (secondCopyStage.fences[currentFrame] == VK_NULL_HANDLE) {
				DEBUG_LOG("Second copy stage fence is null!\n");
			}
			if (secondCopyStage.completeSemaphores[currentFrame] == VK_NULL_HANDLE) {
				DEBUG_LOG("Second copy stage semaphore is null!\n");
			}

			// Submit second copy commands
			VkPipelineStageFlags copyWaitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
			// Third stage (second copy)
			VkSubmitInfo secondCopySubmitInfo{};
			secondCopySubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			secondCopySubmitInfo.commandBufferCount = 1;
			secondCopySubmitInfo.pCommandBuffers = &secondCopyStage.commandBuffers[currentFrame];
			secondCopySubmitInfo.waitSemaphoreCount = 1;
			secondCopySubmitInfo.pWaitSemaphores = &firstCopyStage.completeSemaphores[currentFrame];
			secondCopySubmitInfo.pWaitDstStageMask = copyWaitStages;
			secondCopySubmitInfo.signalSemaphoreCount = 1;
			secondCopySubmitInfo.pSignalSemaphores = &secondCopyStage.completeSemaphores[currentFrame];
			VK_CHECK_RESULT(vkQueueSubmit(queueManager.getQueue(), 1, &secondCopySubmitInfo, secondCopyStage.fences[currentFrame]));
			DEBUG_LOG("Second copy stage submitted\n");
		}

		// Stage 4: Copy RT3 to swapchain
		{
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			VkCommandBuffer cmd = finalCopyStage.commandBuffers[currentFrame];

			VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));

			// Transition RT2 to transfer source
			VkImageMemoryBarrier rt2TransferBarrier{};
			rt2TransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			rt2TransferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			rt2TransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			rt2TransferBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			rt2TransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			rt2TransferBarrier.image = renderTarget2.image;
			rt2TransferBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			// Transition swapchain to transfer destination
			VkImageMemoryBarrier swapchainTransferBarrier{};
			swapchainTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			
			// Continuing with swapchain barrier setup
			swapchainTransferBarrier.srcAccessMask = 0;
			swapchainTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			swapchainTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			swapchainTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			swapchainTransferBarrier.image = swapChain.images[imageIndex];
			swapchainTransferBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			swapchainTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			swapchainTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			VkImageMemoryBarrier barriers[] = { rt2TransferBarrier, swapchainTransferBarrier };
			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				2, barriers);

			// Copy RT2 to swapchain
			VkImageCopy copyRegion{};
			copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.extent = { renderTarget2.width, renderTarget2.height, 1 };

			vkCmdCopyImage(cmd,
				renderTarget2.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				swapChain.images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &copyRegion);

			// Transition swapchain to present layout
			swapchainTransferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			swapchainTransferBarrier.dstAccessMask = 0;
			swapchainTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			swapchainTransferBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			// Transition RT2 back to undefined (will be overwritten next frame)
			rt2TransferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			rt2TransferBarrier.dstAccessMask = 0;
			rt2TransferBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			rt2TransferBarrier.newLayout = VK_IMAGE_LAYOUT_UNDEFINED;

			VkImageMemoryBarrier finalBarriers[] = { swapchainTransferBarrier, rt2TransferBarrier };
			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0,
				0, nullptr,
				0, nullptr,
				2, finalBarriers);

			VK_CHECK_RESULT(vkEndCommandBuffer(cmd));

			// Submit final copy commands
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &cmd;
			submitInfo.waitSemaphoreCount = 1;
			submitInfo.pWaitSemaphores = &secondCopyStage.completeSemaphores[currentFrame];
			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores = &finalCopyStage.completeSemaphores[currentFrame];
			// For final stage (second copy)

			VkPipelineStageFlags copyWaitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
			submitInfo.pWaitDstStageMask = copyWaitStages;  // Reuse the same transfer bit stage flags

			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, finalCopyStage.fences[currentFrame]));
		}

		// Before the present
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapChain.swapChain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &finalCopyStage.completeSemaphores[currentFrame];

		// Handle the result properly
		VkResult result = vkQueuePresentKHR(queue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			// Handle resize/recreation
			windowResize();
			// Don't treat this as an error
			return;
		}
		else if (result != VK_SUCCESS) {
			VK_CHECK_RESULT(result);
		}

		currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
	}
};

// OS specific main entry points
// Most of the code base is shared for the different supported operating systems, but stuff like message handling differs

#if defined(_WIN32)
// Windows entry point
VulkanExample *vulkanExample;
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
VulkanExample *vulkanExample;
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
VulkanExample *vulkanExample;
static void handleEvent()
{
}
int main(const int argc, const char *argv[])
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
VulkanExample *vulkanExample;
static void handleEvent(const DFBWindowEvent *event)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleEvent(event);
	}
}
int main(const int argc, const char *argv[])
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
VulkanExample *vulkanExample;
int main(const int argc, const char *argv[])
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
VulkanExample *vulkanExample;
#if defined(VK_USE_PLATFORM_XCB_KHR)
static void handleEvent(const xcb_generic_event_t *event)
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
int main(const int argc, const char *argv[])
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
VulkanExample *vulkanExample;
int main(const int argc, const char *argv[])
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
