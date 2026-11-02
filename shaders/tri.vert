#version 460
#extension GL_GOOGLE_include_directive : require
#include "draw.glsl"

// Plane w = dot(slope, uv) + c and depth range, in absolute voxel units. Computed only from absolute
// positions, so every pass and every block size sees exactly the same values.
layout(location = 0) flat out uint vTri;
layout(location = 1) flat out uint vAxis;
layout(location = 2) flat out vec2 vSlope;
layout(location = 3) flat out float vC;
layout(location = 4) flat out vec2 vRange;

vec3 vertexPos(uint tri, uint corner) {
    uint i = pc.indices.v[tri * 3u + corner] * 3u;
    return vec3(pc.verts.v[i], pc.verts.v[i + 1u], pc.verts.v[i + 2u]);
}

void main() {
    uint entry = uint(gl_VertexIndex) / 3u;
    uint tri = pc.mode == MODE_BLOCK ? pc.list.v[entry] : entry;
    vec3 p0 = vertexPos(tri, 0u), p1 = vertexPos(tri, 1u), p2 = vertexPos(tri, 2u);

    precise vec3 n = cross(p1 - p0, p2 - p0);
    vec3 an = abs(n);
    uint axis = an.x >= an.y && an.x >= an.z ? 0u : an.y >= an.z ? 1u : 2u;

    vec3 m = swz(n, axis), q0 = swz(p0, axis), q1 = swz(p1, axis), q2 = swz(p2, axis);
    precise vec2 slope = m.z != 0.0 ? -m.xy / m.z : vec2(0.0); // |slope| <= 1 per component
    precise float c = q0.z - dot(slope, q0.xy);

    vTri = tri;
    vAxis = axis;
    vSlope = slope;
    vC = c;
    vRange = vec2(min(q0.z, min(q1.z, q2.z)), max(q0.z, max(q1.z, q2.z)));

    uint corner = uint(gl_VertexIndex) % 3u;
    vec3 p = corner == 0u ? p0 : corner == 1u ? p1 : p2;
    // Exact: origin is a multiple of the block size, pixelSize and size are powers of two. The viewport
    // spans [-2 size, 2 size), so NDC = pixel / (2 size) with no offset to round, and the clip volume ends a
    // whole image away from the drawn area: a triangle just past the image edge still reaches the edge
    // pixels through conservative overestimation instead of being clipped away.
    vec2 pixel = swz(p - pc.origin, axis).xy / pc.pixelSize;
    gl_Position = vec4(pixel / float(2u * pc.size), 0.0, 1.0);
}
