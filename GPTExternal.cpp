VkImage getExternalImage(VkFormat format, uint32_t width, uint32_t height, const char* name) {
	// Step 1: Create an Android Hardware Buffer
	AHardwareBuffer_Desc bufferDesc = {};
	bufferDesc.width = width;
	bufferDesc.height = height;
	bufferDesc.layers = 1;
	bufferDesc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
	bufferDesc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER;

	AHardwareBuffer* hardwareBuffer;
	if (AHardwareBuffer_allocate(&bufferDesc, &hardwareBuffer) != 0) {
		throw std::runtime_error("Failed to allocate AHardwareBuffer");
	}

	// Step 2: Create a Vulkan Image linked to the Hardware Buffer
	VkExternalMemoryImageCreateInfo externalCreateInfo = {};
	externalCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	externalCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

	VkImageCreateInfo imageCreateInfo = {};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.pNext = &externalCreateInfo;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.extent = { width, height, 1 };
	imageCreateInfo.mipLevels = 1;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.format = format;
	imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkImage image;
	VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &image));

	// Step 3: Import the Hardware Buffer into Vulkan
	VkAndroidHardwareBufferPropertiesANDROID bufferProperties = {};
	bufferProperties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;

	PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID =
		reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
			vkGetDeviceProcAddr(device, "vkGetAndroidHardwareBufferPropertiesANDROID"));
	VK_CHECK_RESULT(vkGetAndroidHardwareBufferPropertiesANDROID(device, hardwareBuffer, &bufferProperties));

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, image, &memRequirements);

	VkImportAndroidHardwareBufferInfoANDROID importInfo = {};
	importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
	importInfo.buffer = hardwareBuffer;

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = bufferProperties.allocationSize;
	allocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits, bufferProperties.memoryTypeBits);

	VkDeviceMemory memory;
	allocInfo.pNext = &importInfo;
	VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &memory));

	// Step 4: Bind Memory to Image
	VK_CHECK_RESULT(vkBindImageMemory(device, image, memory, 0));

	// Cleanup the AHardwareBuffer reference if not needed anymore.
	AHardwareBuffer_release(hardwareBuffer);

	return image;
}


VkImage createExternalImage(VkFormat format, uint32_t width, uint32_t height) {
    AHardwareBuffer_Desc bufferDesc = {};
    bufferDesc.width = width;
    bufferDesc.height = height;
    bufferDesc.layers = 1;
    bufferDesc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    bufferDesc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER;

    AHardwareBuffer* hardwareBuffer;
    if (AHardwareBuffer_allocate(&bufferDesc, &hardwareBuffer) != 0) {
        throw std::runtime_error("Failed to allocate AHardwareBuffer");
    }

    VkExternalMemoryImageCreateInfo externalInfo = {};
    externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = { width, height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage externalImage;
    VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &externalImage));

    VkAndroidHardwareBufferPropertiesANDROID bufferProps = {};
    bufferProps.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;

    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAHBProps =
        (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(device, "vkGetAndroidHardwareBufferPropertiesANDROID");
    VK_CHECK_RESULT(vkGetAHBProps(device, hardwareBuffer, &bufferProps));

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, externalImage, &memReqs);

    VkImportAndroidHardwareBufferInfoANDROID importInfo = {};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.buffer = hardwareBuffer;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = bufferProps.allocationSize;
    allocInfo.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    allocInfo.pNext = &importInfo;

    VkDeviceMemory externalMemory;
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &externalMemory));
    VK_CHECK_RESULT(vkBindImageMemory(device, externalImage, externalMemory, 0));

    // Release AHardwareBuffer when no longer needed.
    AHardwareBuffer_release(hardwareBuffer);

    return externalImage;
}

void copyToExternalImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImage dstImage, uint32_t width, uint32_t height) {
    // Transition source and destination image layouts
    transitionImageLayout(commandBuffer, srcImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    transitionImageLayout(commandBuffer, dstImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Define the copy region
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.srcOffset = { 0, 0, 0 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstOffset = { 0, 0, 0 };
    copyRegion.extent = { width, height, 1 };

    vkCmdCopyImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // Transition destination image to shader-readable layout if needed
    transitionImageLayout(commandBuffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void renderToSwapchain(VkCommandBuffer commandBuffer, VkImage externalImage, VkImage swapchainImage, uint32_t width, uint32_t height) {
    // Transition external image to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    transitionImageLayout(commandBuffer, externalImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Transition swapchain image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    transitionImageLayout(commandBuffer, swapchainImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Define the copy region
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.srcOffset = { 0, 0, 0 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstOffset = { 0, 0, 0 };
    copyRegion.extent = { width, height, 1 };

    // Perform the copy
    vkCmdCopyImage(commandBuffer, externalImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // Transition swapchain image to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    transitionImageLayout(commandBuffer, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}


void render() {
    VkResult result = vkWaitForFences(device, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX);
    VK_CHECK_RESULT(result);
    VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentFrame]));

    uint32_t imageIndex;
    result = vkAcquireNextImageKHR(device, swapChain.swapChain, UINT64_MAX,
                                   presentCompleteSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        recreateSwapChain();
        return;
    }
    VK_CHECK_RESULT(result);

    VkCommandBuffer commandBuffer = commandBuffers[currentFrame];
    VK_CHECK_RESULT(vkResetCommandBuffer(commandBuffer, 0));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    // Call renderToSwapchain with the external image and the acquired swapchain image
    renderToSwapchain(commandBuffer, rt3Image, swapChain.images[imageIndex], width, height);

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    submitInfo.pWaitDstStageMask = &waitStage;

    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &presentCompleteSemaphores[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderCompleteSemaphores[currentFrame];
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentFrame]));

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain.swapChain;
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderCompleteSemaphores[currentFrame];

    result = vkQueuePresentKHR(queue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        recreateSwapChain();
    } else {
        VK_CHECK_RESULT(result);
    }

    currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
}
