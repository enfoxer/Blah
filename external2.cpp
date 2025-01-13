VkResult createExternalImage(
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImage& image,
    VkDeviceMemory& memory,
    HANDLE& externalHandle) {
    
    // First check if external memory is supported for this format
    VkPhysicalDeviceImageFormatInfo2 formatInfo{};
    formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    formatInfo.format = format;
    formatInfo.type = VK_IMAGE_TYPE_2D;
    formatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    formatInfo.usage = usage;

    // Configure external memory handle type based on platform
    VkExternalMemoryHandleTypeFlagBits handleType;
    #if defined(_WIN32)
        handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    #elif defined(__ANDROID__)
        handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    #else  // Linux/other platforms
        handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    #endif

    // Check Android-specific requirements if needed
    #ifdef __ANDROID__
    VkAndroidHardwareBufferUsageANDROID ahbUsage{};
    ahbUsage.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID;
    
    VkExternalFormatANDROID externalFormat{};
    externalFormat.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
    externalFormat.externalFormat = 0;  // Will be filled by the driver
    formatInfo.pNext = &externalFormat;
    #endif

    VkPhysicalDeviceExternalImageFormatInfo externalFormatInfo{};
    externalFormatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    #ifdef _WIN32
    externalFormatInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    #else
    externalFormatInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    #endif
    formatInfo.pNext = &externalFormatInfo;

    VkExternalImageFormatProperties externalFormatProps{};
    externalFormatProps.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;

    VkImageFormatProperties2 formatProps{};
    formatProps.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    formatProps.pNext = &externalFormatProps;

    // Query format properties
    VkResult result = vkGetPhysicalDeviceImageFormatProperties2(
        physicalDevice,
        &formatInfo,
        &formatProps
    );

    if (result != VK_SUCCESS) {
        printf("External memory not supported for this format\n");
        return result;
    }

    // Check if export is supported
    if (!(externalFormatProps.externalMemoryProperties.externalMemoryFeatures & 
          VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)) {
        printf("Image format does not support memory export\n");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Create image with external memory support
    VkExternalMemoryImageCreateInfo extImageInfo{};
    extImageInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extImageInfo.handleTypes = handleType;

    #ifdef __ANDROID__
    VkExternalFormatANDROID extFormat{};
    extFormat.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
    extFormat.externalFormat = 0;  // Will be filled by the driver
    
    VkAndroidHardwareBufferUsageANDROID ahbUsage{};
    ahbUsage.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID;
    ahbUsage.pNext = &extFormat;
    extImageInfo.pNext = &ahbUsage;
    #endif

    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.pNext = &extImageInfo;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = format;
    imageCI.extent = {width, height, 1};
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = usage;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    result = vkCreateImage(device, &imageCI, nullptr, &image);
    if (result != VK_SUCCESS) {
        printf("Failed to create image\n");
        return result;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    // Find memory type index
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && 
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    
    if (memTypeIndex == UINT32_MAX) {
        printf("Failed to find suitable memory type\n");
        vkDestroyImage(device, image, nullptr);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Allocate memory with export capability
    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    #ifdef _WIN32
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    #else
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    #endif

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    result = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        printf("Failed to allocate image memory\n");
        vkDestroyImage(device, image, nullptr);
        return result;
    }

    // Bind memory to image
    result = vkBindImageMemory(device, image, memory, 0);
    if (result != VK_SUCCESS) {
        printf("Failed to bind image memory\n");
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return result;
    }

    // Get external handle based on platform
    #if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR getHandleInfo{};
    getHandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getHandleInfo.memory = memory;
    getHandleInfo.handleType = handleType;
    
    auto pfnGetMemoryWin32Handle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!pfnGetMemoryWin32Handle) {
        printf("Failed to get vkGetMemoryWin32HandleKHR function pointer\n");
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    
    result = pfnGetMemoryWin32Handle(device, &getHandleInfo, &externalHandle);

    #elif defined(__ANDROID__)
    VkMemoryGetAndroidHardwareBufferInfoANDROID getAHBInfo{};
    getAHBInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    getAHBInfo.memory = memory;

    AHardwareBuffer* androidHardwareBuffer;
    auto pfnGetMemoryAndroidHardwareBuffer = (PFN_vkGetMemoryAndroidHardwareBufferANDROID)
        vkGetDeviceProcAddr(device, "vkGetMemoryAndroidHardwareBufferANDROID");
    
    if (!pfnGetMemoryAndroidHardwareBuffer) {
        printf("Failed to get vkGetMemoryAndroidHardwareBufferANDROID function pointer\n");
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    result = pfnGetMemoryAndroidHardwareBuffer(device, &getAHBInfo, &androidHardwareBuffer);
    externalHandle = (HANDLE)androidHardwareBuffer;

    #else
    VkMemoryGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = memory;
    getFdInfo.handleType = handleType;

    auto pfnGetMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!pfnGetMemoryFd) {
        printf("Failed to get vkGetMemoryFdKHR function pointer\n");
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    result = pfnGetMemoryFd(device, &getFdInfo, reinterpret_cast<int*>(&externalHandle));
    #endif

    if (result != VK_SUCCESS) {
        printf("Failed to get external handle\n");
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return result;
    }

    return VK_SUCCESS;
}

// Usage example:
/*
VkImage image;
VkDeviceMemory memory;
HANDLE externalHandle;

VkResult result = createExternalImage(
    physicalDevice,
    device,
    width,
    height,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    image,
    memory,
    externalHandle
);

if (result != VK_SUCCESS) {
    // Handle error
}
*/
