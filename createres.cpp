#include <vulkan/vulkan.h>
#include <stdexcept>

struct VulkanImage {
    VkImage image;
    VkDeviceMemory memory;
};

VulkanImage createImage(
    VkDevice device,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageUsageFlags usage,
    VkMemoryPropertyFlags memoryProperties,
    VkPhysicalDevice physicalDevice)
{
    VulkanImage vulkanImage = {};

    // Create the image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &imageInfo, nullptr, &vulkanImage.image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image!");
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, vulkanImage.image, &memRequirements);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    // Find suitable memory type
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & memoryProperties) == memoryProperties) {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX) {
        throw std::runtime_error("Failed to find suitable memory type!");
    }
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(device, &allocInfo, nullptr, &vulkanImage.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory!");
    }

    // Bind memory to the image
    if (vkBindImageMemory(device, vulkanImage.image, vulkanImage.memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind image memory!");
    }

    return vulkanImage;
}



#include <vulkan/vulkan.h>
#include <android/hardware_buffer.h>
#include <stdexcept>

struct VulkanImage {
    VkImage image;
    VkDeviceMemory memory;
};

VulkanImage createExternalImage(
    VkDevice device,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageUsageFlags usage,
    VkPhysicalDevice physicalDevice,
    AHardwareBuffer* hardwareBuffer)
{
    VulkanImage vulkanImage = {};

    // 1. Create Vulkan image with external memory support
    VkExternalMemoryImageCreateInfo externalMemoryInfo{};
    externalMemoryInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalMemoryInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalMemoryInfo; // Link to external memory info
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &imageInfo, nullptr, &vulkanImage.image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create external Vulkan image!");
    }

    // 2. Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, vulkanImage.image, &memRequirements);

    // 3. Import external hardware buffer memory
    VkImportAndroidHardwareBufferInfoANDROID importBufferInfo{};
    importBufferInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importBufferInfo.buffer = hardwareBuffer;

    VkMemoryDedicatedAllocateInfo memoryDedicatedInfo{};
    memoryDedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    memoryDedicatedInfo.image = vulkanImage.image;
    memoryDedicatedInfo.pNext = &importBufferInfo;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &memoryDedicatedInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = 0; // Will be set later.

    // 4. Find suitable memory type for external memory
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    bool memoryTypeFound = false;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            allocInfo.memoryTypeIndex = i;
            memoryTypeFound = true;
            break;
        }
    }

    if (!memoryTypeFound) {
        throw std::runtime_error("Failed to find suitable memory type for external image!");
    }

    if (vkAllocateMemory(device, &allocInfo, nullptr, &vulkanImage.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate external image memory!");
    }

    // 5. Bind memory to the Vulkan image
    if (vkBindImageMemory(device, vulkanImage.image, vulkanImage.memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind external image memory!");
    }

    return vulkanImage;
}





I'll write a function that creates a Vulkan image and allocates/binds memory for it with basic parameters.



```cpp
VkResult createImage(
    VkDevice device,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageTiling tiling,
    VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkImage& image,
    VkDeviceMemory& imageMemory) {
    
    // Image creation info
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Create the image
    VkResult result = vkCreateImage(device, &imageInfo, nullptr, &image);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(device, memRequirements.memoryTypeBits, properties);

    result = vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
    if (result != VK_SUCCESS) {
        vkDestroyImage(device, image, nullptr);
        return result;
    }

    // Bind image memory
    result = vkBindImageMemory(device, image, imageMemory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return result;
    }

    return VK_SUCCESS;
}
```

This function handles:
1. Creating the image with basic parameters
2. Allocating device memory for the image
3. Binding the memory to the image

You can call it like this:

```cpp
VkImage image;
VkDeviceMemory imageMemory;
VkResult result = createImage(
    device,
    width,
    height,
    VK_FORMAT_R8G8B8A8_SRGB,  // Common format for textures
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    image,
    imageMemory
);
```

Note that this assumes you have a `findMemoryType` function that selects the appropriate memory type index. Here's a basic implementation of that helper function:

```cpp
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
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
```

Let me know if you need any clarification or would like to modify any of the parameters!​​​​​​​​​​​​​​​​




I'll modify the image creation function to support external memory using AHardwareBuffer for Android. Here's how to adapt it:

```cpp
VkResult createExternalImage(
    VkDevice device,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageTiling tiling,
    VkImageUsageFlags usage,
    AHardwareBuffer* hardwareBuffer,
    VkImage& image,
    VkDeviceMemory& imageMemory) {
    
    // Get AHardwareBuffer properties
    AHardwareBuffer_Desc ahbDesc;
    AHardwareBuffer_describe(hardwareBuffer, &ahbDesc);

    // External memory image create info
    VkExternalMemoryImageCreateInfo extImageCreateInfo{};
    extImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    // Image creation info
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &extImageCreateInfo;  // Chain the external memory info
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Create the image
    VkResult result = vkCreateImage(device, &imageInfo, nullptr, &image);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Import the AHardwareBuffer
    VkImportAndroidHardwareBufferInfoANDROID importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.buffer = hardwareBuffer;

    // Memory dedicated allocation info
    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{};
    dedicatedAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedAllocInfo.pNext = &importInfo;
    dedicatedAllocInfo.image = image;

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    // Get Android hardware buffer properties
    VkAndroidHardwareBufferPropertiesANDROID hwbProps{};
    hwbProps.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    vkGetAndroidHardwareBufferPropertiesANDROID(device, hardwareBuffer, &hwbProps);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &dedicatedAllocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        device,
        hwbProps.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    result = vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
    if (result != VK_SUCCESS) {
        vkDestroyImage(device, image, nullptr);
        return result;
    }

    // Bind image memory
    result = vkBindImageMemory(device, image, imageMemory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return result;
    }

    return VK_SUCCESS;
}
```

Key differences from the regular image creation:

1. Added `VkExternalMemoryImageCreateInfo` to specify we're using Android hardware buffer
2. Added `VkImportAndroidHardwareBufferInfoANDROID` to import the AHardwareBuffer
3. Used `VkMemoryDedicatedAllocateInfo` for dedicated allocation
4. Used the hardware buffer properties for memory type selection

Usage example:

```cpp
// Assuming you already have your AHardwareBuffer
AHardwareBuffer* hardwareBuffer; // Created elsewhere
VkImage image;
VkDeviceMemory imageMemory;

VkResult result = createExternalImage(
    device,
    width,
    height,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    hardwareBuffer,
    image,
    imageMemory
);

// Don't forget to enable the required extensions:
// VK_ANDROID_external_memory_android_hardware_buffer
// You also need to request the feature in your device creation
```

Remember to:
1. Enable the required device extensions
2. Handle cleanup properly
3. Make sure your AHardwareBuffer format is compatible with your Vulkan format
4. Consider synchronization between Android and Vulkan usage of the buffer

Would you like me to explain any part in more detail?​​​​​​​​​​​​​​​​





