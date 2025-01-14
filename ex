struct VulkanHardwareBufferResources {
    VkImage image{ VK_NULL_HANDLE };
    VkDeviceMemory memory{ VK_NULL_HANDLE };
    VkImageView view{ VK_NULL_HANDLE };
    VkFramebuffer framebuffer{ VK_NULL_HANDLE };
};

VulkanHardwareBufferResources createVulkanResourcesFromHardwareBuffer(
    AHardwareBuffer* hardwareBuffer, 
    VkFormat format,
    VkImageUsageFlags additionalUsage = 0) 
{
    VulkanHardwareBufferResources resources{};
    
    // Get the hardware buffer properties
    VkAndroidHardwareBufferFormatPropertiesANDROID formatInfo{};
    formatInfo.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
    
    VkAndroidHardwareBufferPropertiesANDROID bufferProperties{};
    bufferProperties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    bufferProperties.pNext = &formatInfo;
    
    auto fpGetAndroidHardwareBufferProperties = 
        (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(
            device, "vkGetAndroidHardwareBufferPropertiesANDROID");
    VK_CHECK_RESULT(fpGetAndroidHardwareBufferProperties(
        device, hardwareBuffer, &bufferProperties));

    // Get the hardware buffer dimensions
    AHardwareBuffer_Desc bufferDesc;
    AHardwareBuffer_describe(hardwareBuffer, &bufferDesc);

    // Create the Vulkan image
    VkExternalFormatANDROID externalFormat{};
    externalFormat.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
    externalFormat.externalFormat = formatInfo.externalFormat;

    VkExternalMemoryImageCreateInfo externalCreateInfo{};
    externalCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalCreateInfo.handleTypes = 
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    externalCreateInfo.pNext = &externalFormat;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalCreateInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = { bufferDesc.width, bufferDesc.height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | 
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | 
                     additionalUsage;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    if (externalFormat.externalFormat != 0) {
        imageInfo.format = VK_FORMAT_UNDEFINED;
    }

    VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &resources.image));

    // Allocate and bind memory
    VkImportAndroidHardwareBufferInfoANDROID importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.buffer = hardwareBuffer;

    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{};
    dedicatedAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedAllocInfo.pNext = &importInfo;
    dedicatedAllocInfo.image = resources.image;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &dedicatedAllocInfo;
    allocInfo.allocationSize = bufferProperties.allocationSize;
    allocInfo.memoryTypeIndex = getMemoryTypeIndex(
        bufferProperties.memoryTypeBits, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &resources.memory));
    VK_CHECK_RESULT(vkBindImageMemory(device, resources.image, resources.memory, 0));

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.image = resources.image;
    viewInfo.subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT,
        0, 1,  // mip levels
        0, 1   // array layers
    };
    
    VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &resources.view));

    // Create framebuffer
    std::array<VkImageView, 2> attachments = {
        resources.view,
        depthStencil.view  // Using the existing depth buffer
    };

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;  // Using the existing render pass
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = bufferDesc.width;
    framebufferInfo.height = bufferDesc.height;
    framebufferInfo.layers = 1;

    VK_CHECK_RESULT(vkCreateFramebuffer(
        device, &framebufferInfo, nullptr, &resources.framebuffer));

    return resources;
}

void transitionHardwareBufferImage(
    VkCommandBuffer cmdBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkImageAspectFlags aspectMask)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspectMask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else {
        throw std::runtime_error("Unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        cmdBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void blitToHardwareBuffer(
    VkCommandBuffer cmdBuffer, 
    VkImage sourceImage, 
    VulkanHardwareBufferResources& targetResources,
    uint32_t width,
    uint32_t height)
{
    // Transition source image to transfer source layout
    transitionHardwareBufferImage(
        cmdBuffer,
        sourceImage,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    // Transition target image to transfer destination layout
    transitionHardwareBufferImage(
        cmdBuffer,
        targetResources.image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    // Perform the blit operation
    VkImageBlit blit{};
    blit.srcOffsets[0] = { 0, 0, 0 };
    blit.srcOffsets[1] = { (int32_t)width, (int32_t)height, 1 };
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = 0;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.dstOffsets[0] = { 0, 0, 0 };
    blit.dstOffsets[1] = { (int32_t)width, (int32_t)height, 1 };
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = 0;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;

    vkCmdBlitImage(
        cmdBuffer,
        sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        targetResources.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit,
        VK_FILTER_LINEAR
    );

    // Transition images back to color attachment optimal
    transitionHardwareBufferImage(
        cmdBuffer,
        sourceImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    transitionHardwareBufferImage(
        cmdBuffer,
        targetResources.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
}

void cleanupVulkanHardwareBufferResources(VulkanHardwareBufferResources& resources) {
    if (resources.framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, resources.framebuffer, nullptr);
    }
    if (resources.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, resources.view, nullptr);
    }
    if (resources.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, resources.image, nullptr);
    }
    if (resources.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, resources.memory, nullptr);
    }
    resources = VulkanHardwareBufferResources{};
}




class VulkanExample : public VulkanExampleBase {
public:
    // ... other existing members ...

    // Replace secondaryRenderTarget with:
    AHardwareBuffer* hardwareBuffer{ nullptr };
    VulkanHardwareBufferResources secondaryResources{};
    VkImageLayout currentHardwareBufferLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    void prepare() {
        VulkanExampleBase::prepare();
        createSynchronizationPrimitives();
        createCommandBuffers();

        // Create primary render target as before
        createRenderTarget(primaryRenderTarget);

        // Create hardware buffer and Vulkan resources
        hardwareBuffer = CreateHardwareBuffer(1280, 800, AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
            AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE);
        
        secondaryResources = createVulkanResourcesFromHardwareBuffer(
            hardwareBuffer, 
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT
        );

        // Initialize hardware buffer layout
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands();
        transitionHardwareBufferImage(
            cmdBuffer,
            secondaryResources.image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        endSingleTimeCommands(cmdBuffer);
        currentHardwareBufferLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        createVertexBuffer();
        createUniformBuffers();
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSets();
        createPipelines();
        prepared = true;
    }

    void render() {
        if (!prepared)
            return;

        // Wait for fence and reset
        vkWaitForFences(device, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX);
        VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentFrame]));

        // Update uniform buffer
        ShaderData shaderData{};
        shaderData.projectionMatrix = camera.matrices.perspective;
        shaderData.viewMatrix = camera.matrices.view;
        shaderData.modelMatrix = glm::mat4(1.0f);
        memcpy(uniformBuffers[currentFrame].mapped, &shaderData, sizeof(ShaderData));

        VkCommandBuffer commandBuffer = commandBuffers[currentFrame];
        VkCommandBufferBeginInfo cmdBufInfo{};
        cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

        // First render pass: Render to primary render target
        VkClearValue clearValues[2]{};
        clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset = { 0, 0 };
        renderPassBeginInfo.renderArea.extent = { primaryRenderTarget.width, primaryRenderTarget.height };
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;
        renderPassBeginInfo.framebuffer = primaryRenderTarget.framebuffer;

        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // ... Rest of your rendering code ...

        vkCmdEndRenderPass(commandBuffer);

        // Blit primary render target to hardware buffer
        blitToHardwareBuffer(
            commandBuffer,
            primaryRenderTarget.image,
            secondaryResources,
            primaryRenderTarget.width,
            primaryRenderTarget.height
        );

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

        // Submit command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.waitSemaphoreCount = 0;

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentFrame]));

        // Keep window system happy
        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapChain.swapChain, UINT64_MAX, presentCompleteSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain.swapChain;
        presentInfo.pImageIndices = &imageIndex;
        vkQueuePresentKHR(queue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
    }

    ~VulkanExample() {
        // Cleanup Vulkan resources
        cleanupVulkanHardwareBufferResources(secondaryResources);
        
        // Note: Don't cleanup hardwareBuffer here if it's managed elsewhere
        // If you do need to cleanup here, add:
        // if (hardwareBuffer) {
        //     AHardwareBuffer_release(hardwareBuffer);
        // }

        // ... rest of cleanup ...
    }
};
