#version 460
#extension GL_GOOGLE_include_directive : require
#include "draw.glsl"

layout(location = 0) flat in uint vTri;
layout(location = 1) flat in uint vAxis;
layout(location = 2) flat in vec2 vSlope;
layout(location = 3) flat in float vC;
layout(location = 4) flat in vec2 vRange;

void emit(ivec3 block) {
    if (any(lessThan(block, ivec3(0))) || any(greaterThanEqual(block, ivec3(pc.size)))) return;
    uint b = morton3(uvec3(block));
    if (pc.mode == MODE_COUNT) {
        atomicAdd(pc.counters.v[b], 1u);
        return;
    }
    if (b < pc.first || b > pc.last) return;
    uint i = atomicAdd(pc.counters.v[b], 0xFFFFFFFFu) - 1u; // counter holds the end of the block's range
    pc.list.v[i] = vTri;
}

void main() {
    // One pixel = one block column. Depth over the pixel spans center +- h, limited to the triangle's own
    // depth range. eps absorbs float differences against the block pass.
    const float eps = 2.0;
    vec2 uv = gl_FragCoord.xy * pc.pixelSize;
    float zc = dot(vSlope, uv) + vC;
    float h = 0.5 * pc.pixelSize * (abs(vSlope.x) + abs(vSlope.y));
    int lo = int(floor((clamp(zc - h, vRange.x, vRange.y) - eps) / pc.pixelSize));
    int hi = int(floor((clamp(zc + h, vRange.x, vRange.y) + eps) / pc.pixelSize));
    // The block at the clamped center depth is always in [lo, hi]; slope <= 1 limits the rest to the
    // blocks directly before and after it.
    for (int w = lo; w <= hi; ++w)
        emit(unswz(ivec3(ivec2(gl_FragCoord.xy), w), vAxis));
}
