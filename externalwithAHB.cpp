#include <vulkan/vulkan.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>

VkResult importImageFromAHardwareBuffer(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    AHardwareBuffer* hardwareBuffer,
    VkImage& outImage,
    VkDeviceMemory& outMemory) {
    
    // Get AHardwareBuffer properties
    AHardwareBuffer_Desc ahbDesc;
    AHardwareBuffer_describe(hardwareBuffer, &ahbDesc);

    // Get Android Hardware Buffer properties
    VkAndroidHardwareBufferFormatPropertiesANDROID formatInfo{};
    formatInfo.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
    
    VkAndroidHardwareBufferPropertiesANDROID properties{};
    properties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    properties.pNext = &formatInfo;

    PFN_vkGetAndroidHardwareBufferPropertiesANDROID pfnGetAndroidHardwareBufferProperties =
        (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(device, "vkGetAndroidHardwareBufferPropertiesANDROID");
    if (!pfnGetAndroidHardwareBufferProperties) {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    VkResult result = pfnGetAndroidHardwareBufferProperties(device, hardwareBuffer, &properties);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Create Image
    VkExternalMemoryImageCreateInfo extImageCreateInfo{};
    extImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.pNext = &extImageCreateInfo;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = formatInfo.format;
    imageCreateInfo.extent = { ahbDesc.width, ahbDesc.height, 1 };
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;  // Adjust based on your needs
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    result = vkCreateImage(device, &imageCreateInfo, nullptr, &outImage);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Allocate and bind memory
    VkImportAndroidHardwareBufferInfoANDROID importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.buffer = hardwareBuffer;

    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.pNext = &importInfo;
    dedicatedInfo.image = outImage;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &dedicatedInfo;
    allocInfo.allocationSize = properties.allocationSize;

    // Find proper memory type
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, outImage, &memReqs);
    
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) && 
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            allocInfo.memoryTypeIndex = i;
            break;
        }
    }

    result = vkAllocateMemory(device, &allocInfo, nullptr, &outMemory);
    if (result != VK_SUCCESS) {
        vkDestroyImage(device, outImage, nullptr);
        return result;
    }

    result = vkBindImageMemory(device, outImage, outMemory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device, outMemory, nullptr);
        vkDestroyImage(device, outImage, nullptr);
        return result;
    }

    return VK_SUCCESS;
}

// Usage example:
/*
AHardwareBuffer* hardwareBuffer = ...; // Your existing AHardwareBuffer
VkImage image;
VkDeviceMemory memory;

VkResult result = importImageFromAHardwareBuffer(
    device,
    physicalDevice,
    hardwareBuffer,
    image,
    memory
);

if (result != VK_SUCCESS) {
    // Handle error
}
*/