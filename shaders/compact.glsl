// Shared by the compaction steps. A compaction group is 256 consecutive bricks of the dense block.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 8) buffer Bricks { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Uints { uint v[]; };
struct Record { uvec4 v; }; // key lo/hi, mask lo/hi: one 16-byte store per record over PCIe
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Records { uint64_t count; uint64_t pad; Record r[]; };

// Must match CompactPC in vk.hpp.
layout(push_constant) uniform PC {
    Bricks dense;       // dense block, bricks in Morton order
    Uints counts;       // non-empty bricks per group, counted by voxel.frag
    Uints offsets;      // per group: output offset within its run of 1024 groups
    Uints totals;       // per run of 1024 groups: record count, then (after scan) output offset
    Records records;    // compact buffer in host memory
    uint64_t keyPrefix; // Morton of the block, shifted above the brick bits
    uint brickCount;
    uint groupCount;
} pc;

// Exclusive prefix sum of one value per thread over the whole workgroup (1024 threads). Returns the
// thread's exclusive prefix; `total` gets the workgroup sum.
shared uint subgroupSums[257]; // up to 256 subgroups (size >= 4) plus the total
uint workgroupExclusiveSum(uint value, out uint total) {
    uint inclusive = subgroupInclusiveAdd(value);
    if (gl_SubgroupInvocationID == gl_SubgroupSize - 1u) subgroupSums[gl_SubgroupID] = inclusive;
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        uint acc = 0u;
        for (uint s = 0u; s < gl_NumSubgroups; ++s) {
            uint x = subgroupSums[s];
            subgroupSums[s] = acc;
            acc += x;
        }
        subgroupSums[gl_NumSubgroups] = acc;
    }
    barrier();
    total = subgroupSums[gl_NumSubgroups];
    return subgroupSums[gl_SubgroupID] + inclusive - value;
}
