// Shared by tri.vert, bin.frag and voxel.frag. All coordinates are absolute voxel units unless noted.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Uints { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Bricks { uint64_t v[]; };

const uint MODE_COUNT = 0u; // binning: count entries per block
const uint MODE_FILL = 1u;  // binning: write triangle ids into the list
const uint MODE_BLOCK = 2u; // full-resolution rasterization of one block

// Must match DrawPC in vk.hpp.
layout(push_constant) uniform PC {
    Floats verts;    // xyz per vertex, voxel units
    Uints indices;   // 3 per triangle
    Uints list;      // triangle list of the current batch
    Uints counters;  // binning: per-block counters (Morton index)
    vec3 origin;     // block origin (0 for binning)
    float pixelSize; // voxels per pixel: block size for binning, 1 for blocks
    uint size;       // image size in pixels
    uint mode;
    uint first;      // FILL: Morton range of blocks in the batch
    uint last;
    Bricks dense;    // block: brick masks in Morton order
    Uints groupCounts; // block: non-empty bricks per compaction group (256 bricks)
} pc;

// Projection along the dominant axis: (u, v) are the image axes, w is depth.
vec3 swz(vec3 p, uint axis) { return axis == 0u ? p.yzx : axis == 1u ? p.zxy : p; }
ivec3 unswz(ivec3 q, uint axis) { return axis == 0u ? q.zxy : axis == 1u ? q.yzx : q; }

uint part1by2(uint x) {
    x &= 0x3FFu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}
uint morton3(uvec3 p) { return part1by2(p.x) | (part1by2(p.y) << 1) | (part1by2(p.z) << 2); }
