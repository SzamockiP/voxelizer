#version 460
#extension GL_GOOGLE_include_directive : require
#include "draw.glsl"

layout(location = 0) flat in uint vTri;
layout(location = 1) flat in uint vAxis;
layout(location = 2) flat in vec2 vSlope;
layout(location = 3) flat in float vC;
layout(location = 4) flat in vec2 vRange;

void main() {
    // One pixel = one voxel column; mark the voxel at the depth of the pixel center.
    vec3 o = swz(pc.origin, vAxis);
    precise float z = clamp(dot(vSlope, gl_FragCoord.xy + o.xy) + vC, vRange.x, vRange.y);
    int w = int(floor(z)) - int(o.z);
    if (w < 0 || w >= int(pc.size)) return; // other block: redundant triangle from binning

    uvec3 p = uvec3(unswz(ivec3(ivec2(gl_FragCoord.xy), w), vAxis));
    uint brick = morton3(p >> 2u);
    uint bit = (p.x & 3u) | ((p.y & 3u) << 2) | ((p.z & 3u) << 4);
    // Exactly one fragment sees the brick go from empty to non-empty; it counts the brick for compaction.
    if (atomicOr(pc.dense.v[brick], uint64_t(1) << bit) == 0ul)
        atomicAdd(pc.groupCounts.v[brick >> 8], 1u);
}
