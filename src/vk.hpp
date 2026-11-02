#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstddef>
#include <cstdint>
#include <vector>

void vkCheck(VkResult result, const char* call);
#define VK_CHECK(call) vkCheck((call), #call)

// Push constants of tri.vert / bin.frag / voxel.frag (see shaders/draw.glsl).
struct DrawPC {
    VkDeviceAddress verts, indices, list, counters;
    float origin[3];
    float pixelSize;
    uint32_t size, mode, first, last;
    VkDeviceAddress dense, groupCounts;
};
// std430 offsets in draw.glsl; vec3 origin is 16-aligned and pixelSize packs into its last 4 bytes.
static_assert(offsetof(DrawPC, origin) == 32 && offsetof(DrawPC, pixelSize) == 44 && offsetof(DrawPC, size) == 48 &&
              offsetof(DrawPC, dense) == 64 && sizeof(DrawPC) == 80);
enum : uint32_t { MODE_COUNT = 0, MODE_FILL = 1, MODE_BLOCK = 2 };

// Push constants of the compaction shaders (see shaders/compact.glsl).
struct CompactPC {
    VkDeviceAddress dense, counts, offsets, totals, records;
    uint64_t keyPrefix;
    uint32_t brickCount, groupCount;
};
static_assert(offsetof(CompactPC, keyPrefix) == 40 && offsetof(CompactPC, brickCount) == 48 && sizeof(CompactPC) == 56);

enum class Memory {
    Device,     // VRAM only
    Bar,        // VRAM mapped into the CPU address space (ReBAR), coherent
    HostCached, // system RAM, cached and coherent for fast CPU access; GPU accesses it over PCIe
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkDeviceAddress address = 0;
    void* mapped = nullptr;
};

// Everything Vulkan: device with all required features, pipelines, one queue with a timeline semaphore.
// Owns every buffer it creates; all are freed on destruction (after the device is idle).
class Gpu {
public:
    explicit Gpu(uint32_t maxImageSize);
    ~Gpu();
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;

    Buffer createBuffer(const char* name, VkDeviceSize size, Memory memory);
    VkCommandBuffer createCommandBuffer();
    void begin(VkCommandBuffer cmd);
    void end(VkCommandBuffer cmd);
    // Global memory barrier: all earlier writes (GPU and transfers) visible to all later GPU work and the host.
    void barrier(VkCommandBuffer cmd);
    void draw(VkCommandBuffer cmd, VkPipeline pipeline, const DrawPC& pc, uint32_t vertexCount, uint32_t firstVertex);
    void dispatch(VkCommandBuffer cmd, VkPipeline pipeline, const CompactPC& pc, uint32_t groups);
    // Returns the timeline value reached when the command buffer completes. Producer thread only.
    uint64_t submit(VkCommandBuffer cmd);
    void wait(uint64_t value);
    uint64_t lastSubmitted() const { return value_; }

    VkPipeline binPipeline = VK_NULL_HANDLE, voxelPipeline = VK_NULL_HANDLE;
    VkPipeline scanGroupsPipeline = VK_NULL_HANDLE, scanTotalsPipeline = VK_NULL_HANDLE, writePipeline = VK_NULL_HANDLE;

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t family_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    uint64_t value_ = 0; // last value a submission will signal
    VmaAllocator vma_ = nullptr;
    VkPipelineLayout drawLayout_ = VK_NULL_HANDLE, compactLayout_ = VK_NULL_HANDLE;
    std::vector<Buffer> buffers_;
};
