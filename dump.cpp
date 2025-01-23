#define VK_USE_PLATFORM_WIN32_KHR  // Use appropriate platform macro
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <vector>
#include <iostream>

VkResult DumpRenderTarget(
    VkDevice device,
    VkQueue queue,
    VkImage sourceImage,
    VkFormat imageFormat,
    uint32_t width,
    uint32_t height,
    const char* filename,
    uint32_t queueFamilyIndex) {
    
    // First verify that the device supports the copy command
    VkPhysicalDevice physicalDevice; // You'll need to pass this or query it from the device
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

    // Create command pool for transfer commands
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndex;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create command pool!" << std::endl;
        return result;
    }

    // Create command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    result = vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        std::cerr << "Failed to allocate command buffer!" << std::endl;
        return result;
    }

    // Calculate proper buffer size based on format
    VkDeviceSize pixelSize = 4; // Assuming RGBA8 format
    switch (imageFormat) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_UNORM:
            pixelSize = 4;
            break;
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_B8G8R8_UNORM:
            pixelSize = 3;
            break;
        // Add other formats as needed
    }
    VkDeviceSize imageSize = width * height * pixelSize;

    // Create staging buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    result = vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer);
    if (result != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkDestroyCommandPool(device, commandPool, nullptr);
        std::cerr << "Failed to create staging buffer!" << std::endl;
        return result;
    }

    // Get memory requirements and allocate
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memRequirements.size;
    allocateInfo.memoryTypeIndex = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    result = vkAllocateMemory(device, &allocateInfo, nullptr, &stagingMemory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkDestroyCommandPool(device, commandPool, nullptr);
        std::cerr << "Failed to allocate staging memory!" << std::endl;
        return result;
    }

    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

    // Begin command buffer recording
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to begin command buffer!" << std::endl;
        // Cleanup and return
        return result;
    }

    // Transition image layout for transfer
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = sourceImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    // Set up and execute the copy command
    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;   // Tightly packed
    copyRegion.bufferImageHeight = 0;  // Tightly packed
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {width, height, 1};

    vkCmdCopyImageToBuffer(
        commandBuffer,
        sourceImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        stagingBuffer,
        1,
        &copyRegion
    );

    // Transition image layout back
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to end command buffer!" << std::endl;
        // Cleanup and return
        return result;
    }

    // Submit command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Create fence for synchronization
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    
    VkFence fence = VK_NULL_HANDLE;
    result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create fence!" << std::endl;
        // Cleanup and return
        return result;
    }

    result = vkQueueSubmit(queue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to submit queue!" << std::endl;
        vkDestroyFence(device, fence, nullptr);
        // Cleanup and return
        return result;
    }

    // Wait for the command buffer to finish executing
    result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to wait for fence!" << std::endl;
        // Cleanup and return
        return result;
    }

    // Map memory and save to file
    void* data;
    result = vkMapMemory(device, stagingMemory, 0, imageSize, 0, &data);
    if (result == VK_SUCCESS) {
        // Save the data to a file
        FILE* fp = fopen(filename, "wb");
        if (fp) {
            fwrite(data, 1, imageSize, fp);
            fclose(fp);
        }
        vkUnmapMemory(device, stagingMemory);
    }

    // Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    return result;
}

// Helper function to find suitable memory type
uint32_t FindMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

// Example usage:
/*
void CaptureRenderTarget() {
    // After your regular vkQueueSubmit call
    VkResult result = DumpRenderTarget(
        device,                     // Your VkDevice
        queue,                      // Your VkQueue
        swapchainImage,            // Your render target VkImage
        VK_FORMAT_B8G8R8A8_UNORM,  // Your image format
        width,                      // Render target width
        height,                     // Render target height
        "render_target_dump.raw",   // Output filename
        queueFamilyIndex           // Your queue family index
    );

    if (result != VK_SUCCESS) {
        // Handle error
    }
}
*/