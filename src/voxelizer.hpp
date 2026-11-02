#pragma once
#include <cstdint>
#include <filesystem>
#include <vector>

struct Mesh {
    std::vector<float> xyz;       // vertex positions in voxel units, inside [0, grid)
    std::vector<uint32_t> indices; // 3 per triangle
};

struct Params {
    uint32_t grid = 0;                // voxels per axis, power of two
    uint32_t block = 1024;            // voxels per axis, power of two
    uint64_t listBudget = 1ull << 30; // bytes for the triangle list of one batch
    uint32_t buffers = 4;             // compact buffers in flight
};

// One output record: brick Morton key and 64-bit brick mask.
constexpr uint64_t kRecordBytes = 16;

struct Stats {
    uint64_t batches = 0, blocks = 0, bricks = 0;
};

// Writes (Morton key, brick mask) pairs, sorted by key, to `out`.
Stats voxelize(const Mesh& mesh, const Params& params, const std::filesystem::path& out);
