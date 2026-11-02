#define VMA_IMPLEMENTATION
#include "vk.hpp"

#include <vulkan/vk_enum_string_helper.h>

#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

const uint32_t kTriVert[] = {
#include "tri.vert.inc"
};
const uint32_t kBinFrag[] = {
#include "bin.frag.inc"
};
const uint32_t kVoxelFrag[] = {
#include "voxel.frag.inc"
};
const uint32_t kCompactScanGroups[] = {
#include "compact_scan_groups.comp.inc"
};
const uint32_t kCompactScanTotals[] = {
#include "compact_scan_totals.comp.inc"
};
const uint32_t kCompactWrite[] = {
#include "compact_write.comp.inc"
};

// Binning grows triangles by this many block pixels on top of conservative rasterization, so vertex
// snapping at block resolution can never lose a block that the voxel pass touches.
constexpr float kBinOverestimation = 0.25f;

// Empty string if the device has everything; otherwise a list of what is missing.
std::string missingRequirements(VkPhysicalDevice device, uint32_t maxImageSize, uint32_t& queueFamily) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    if (props.apiVersion < VK_API_VERSION_1_3) return " Vulkan 1.3;";

    std::string missing;
    auto need = [&](bool ok, const char* what) {
        if (!ok) missing += std::format(" {};", what);
    };

    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
    auto hasExtension = [&](std::string_view name) {
        for (const auto& e : extensions)
            if (name == e.extensionName) return true;
        return false;
    };
    bool conservative = hasExtension(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
    need(conservative, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);

    // Extension structs are chained only when the extension exists.
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT cons{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    if (conservative) cons.pNext = std::exchange(p11.pNext, &cons);
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p11};
    vkGetPhysicalDeviceProperties2(device, &props2);

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    vkGetPhysicalDeviceFeatures2(device, &f);

    need(f.features.fragmentStoresAndAtomics, "fragmentStoresAndAtomics");
    need(f.features.shaderInt64, "shaderInt64");
    need(f12.timelineSemaphore, "timelineSemaphore");
    need(f12.bufferDeviceAddress, "bufferDeviceAddress");
    need(f12.shaderBufferInt64Atomics, "shaderBufferInt64Atomics");
    need(f13.synchronization2, "synchronization2");
    need(f13.dynamicRendering, "dynamicRendering");
    need(p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT, "subgroup operations in compute shaders");
    const VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    need((p11.subgroupSupportedOperations & ops) == ops, "subgroup basic/ballot/arithmetic operations");
    need(p11.subgroupSize >= 4, "subgroup size >= 4");
    if (conservative)
        need(cons.maxExtraPrimitiveOverestimationSize >= kBinOverestimation, "maxExtraPrimitiveOverestimationSize >= 0.25");
    const auto& limits = props.limits;
    need(limits.maxFramebufferWidth >= maxImageSize && limits.maxFramebufferHeight >= maxImageSize,
         "maxFramebufferWidth/Height >= image size");
    need(limits.maxViewportDimensions[0] >= 4 * maxImageSize && limits.maxViewportDimensions[1] >= 4 * maxImageSize,
         "maxViewportDimensions >= 4 * image size");
    need(limits.viewportBoundsRange[0] <= -2.0f * maxImageSize, "viewportBoundsRange reaching -2 * image size");

    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        VkQueueFlags flags = families[i].queueFlags;
        if (queueFamily == UINT32_MAX && (flags & VK_QUEUE_GRAPHICS_BIT) && (flags & VK_QUEUE_COMPUTE_BIT)) queueFamily = i;
    }
    need(queueFamily != UINT32_MAX, "graphics+compute queue");
    return missing;
}

VkShaderModule createModule(VkDevice device, const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes;
    info.pCode = code;
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

VkPipeline createDrawPipeline(VkDevice device, VkPipelineLayout layout, const uint32_t* frag, size_t fragBytes, float overestimation) {
    VkShaderModule vs = createModule(device, kTriVert, sizeof(kTriVert));
    VkShaderModule fs = createModule(device, frag, fragBytes);
    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"},
    };
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservative{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT};
    conservative.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
    conservative.extraPrimitiveOverestimationSize = overestimation;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, &conservative};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO}; // no attachments

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
    return pipeline;
}

VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, const uint32_t* code, size_t bytes) {
    VkShaderModule module = createModule(device, code, bytes);
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main"};
    info.layout = layout;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

VkPipelineLayout createLayout(VkDevice device, VkShaderStageFlags stages, uint32_t pushSize) {
    VkPushConstantRange range{stages, 0, pushSize};
    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &info, nullptr, &layout));
    return layout;
}

} // namespace

void vkCheck(VkResult result, const char* call) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::format("{} failed: {}", call, string_VkResult(result)));
}

Gpu::Gpu(uint32_t maxImageSize) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "voxelizer";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance_));

    // First suitable device, discrete preferred.
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
    std::string report;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        uint32_t family;
        std::string missing = missingRequirements(device, maxImageSize, family);
        if (!missing.empty()) {
            report += std::format("\n  {}: missing{}", props.deviceName, missing);
            continue;
        }
        if (!physical_ || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_ = device;
            family_ = family;
        }
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
    }
    if (!physical_) throw std::runtime_error("no Vulkan device meets the requirements:" + report);

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.synchronization2 = VK_TRUE;
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = VK_TRUE;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderBufferInt64Atomics = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    features.features.fragmentStoresAndAtomics = VK_TRUE;
    features.features.shaderInt64 = VK_TRUE;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family_, 1, &priority};
    const char* extensions[]{VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME};
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = 1;
    deviceInfo.ppEnabledExtensionNames = extensions;
    VK_CHECK(vkCreateDevice(physical_, &deviceInfo, nullptr, &device_));
    vkGetDeviceQueue(device_, family_, 0, &queue_);

    VmaAllocatorCreateInfo vmaInfo{};
    vmaInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaInfo.instance = instance_;
    vmaInfo.physicalDevice = physical_;
    vmaInfo.device = device_;
    VK_CHECK(vmaCreateAllocator(&vmaInfo, &vma_));

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = family_;
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &pool_));

    VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timelineInfo};
    VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &timeline_));

    drawLayout_ = createLayout(device_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(DrawPC));
    compactLayout_ = createLayout(device_, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(CompactPC));
    binPipeline = createDrawPipeline(device_, drawLayout_, kBinFrag, sizeof(kBinFrag), kBinOverestimation);
    voxelPipeline = createDrawPipeline(device_, drawLayout_, kVoxelFrag, sizeof(kVoxelFrag), 0.0f);
    scanGroupsPipeline = createComputePipeline(device_, compactLayout_, kCompactScanGroups, sizeof(kCompactScanGroups));
    scanTotalsPipeline = createComputePipeline(device_, compactLayout_, kCompactScanTotals, sizeof(kCompactScanTotals));
    writePipeline = createComputePipeline(device_, compactLayout_, kCompactWrite, sizeof(kCompactWrite));
}

Gpu::~Gpu() {
    vkDeviceWaitIdle(device_);
    for (const Buffer& b : buffers_) vmaDestroyBuffer(vma_, b.buffer, b.allocation);
    for (VkPipeline p : {binPipeline, voxelPipeline, scanGroupsPipeline, scanTotalsPipeline, writePipeline})
        vkDestroyPipeline(device_, p, nullptr);
    vkDestroyPipelineLayout(device_, drawLayout_, nullptr);
    vkDestroyPipelineLayout(device_, compactLayout_, nullptr);
    vkDestroySemaphore(device_, timeline_, nullptr);
    vkDestroyCommandPool(device_, pool_, nullptr);
    vmaDestroyAllocator(vma_);
    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}

Buffer Gpu::createBuffer(const char* name, VkDeviceSize size, Memory memory) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    switch (memory) {
    case Memory::Device:
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;
    case Memory::Bar:
        allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case Memory::HostCached:
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    }
    Buffer b;
    VmaAllocationInfo info;
    VkResult result = vmaCreateBuffer(vma_, &bufferInfo, &allocInfo, &b.buffer, &b.allocation, &info);
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::format("cannot allocate {} ({} MiB{}): {}", name, (size + (1 << 20) - 1) >> 20,
                                             memory == Memory::Bar ? ", device-local host-visible memory (ReBAR)" : "",
                                             string_VkResult(result)));
    b.mapped = info.pMappedData;
    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(device_, &addressInfo);
    buffers_.push_back(b);
    return b;
}

VkCommandBuffer Gpu::createCommandBuffer() {
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = pool_;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device_, &info, &cmd));
    return cmd;
}

void Gpu::begin(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &info));
}

void Gpu::end(VkCommandBuffer cmd) { VK_CHECK(vkEndCommandBuffer(cmd)); }

void Gpu::barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void Gpu::draw(VkCommandBuffer cmd, VkPipeline pipeline, const DrawPC& pc, uint32_t vertexCount, uint32_t firstVertex) {
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea = {{0, 0}, {pc.size, pc.size}};
    rendering.layerCount = 1;
    vkCmdBeginRendering(cmd, &rendering);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    // Viewport [-2 size, 2 size): pixel = NDC * 2 size, exactly; only [0, size) is drawn (see tri.vert).
    float s = float(pc.size);
    VkViewport viewport{-2 * s, -2 * s, 4 * s, 4 * s, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, {pc.size, pc.size}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdPushConstants(cmd, drawLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd, vertexCount, 1, firstVertex, 0);
    vkCmdEndRendering(cmd);
}

void Gpu::dispatch(VkCommandBuffer cmd, VkPipeline pipeline, const CompactPC& pc, uint32_t groups) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd, compactLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups, 1, 1);
}

uint64_t Gpu::submit(VkCommandBuffer cmd) {
    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = cmd;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = timeline_;
    signal.value = value_ + 1;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    VK_CHECK(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE));
    return ++value_;
}

void Gpu::wait(uint64_t value) {
    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    info.semaphoreCount = 1;
    info.pSemaphores = &timeline_;
    info.pValues = &value;
    VK_CHECK(vkWaitSemaphores(device_, &info, UINT64_MAX));
}
