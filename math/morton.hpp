#pragma once

#include <cstdint>

namespace ic_voxel::math
{
    // 64-bit Morton code for global spatial sorting
    using MortonCode = std::uint64_t;

    // Expands a 21-bit integer into 64 bits by inserting 2 zeros after each bit.
    // This is used for generating Z-order curves (Morton codes).
    [[nodiscard]] inline MortonCode split_by_3(std::uint64_t a)
    {
        a &= 0x1ffffff;
        a = (a | a << 32) & 0x001f00000000ffff;
        a = (a | a << 16) & 0x001f0000ff0000ff;
        a = (a | a << 8 ) & 0x100f00f00f00f00f;
        a = (a | a << 4 ) & 0x10c30c30c30c30c3;
        a = (a | a << 2 ) & 0x1249249249249249;
        return a;
    }

    // Interleaves the bits of 3 coordinates to produce a single 64-bit Morton code.
    [[nodiscard]] inline MortonCode encode_morton(std::uint32_t x, std::uint32_t y, std::uint32_t z)
    {
        return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2);
    }

} // namespace ic_voxel::math
