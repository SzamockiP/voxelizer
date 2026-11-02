#include "voxelizer.hpp"

#include <tiny_obj_loader.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>

namespace {

const char* kUsage =
    "usage: voxelizer <model.obj> <out.bin> --grid N [--block 1024] [--list-budget 1G] [--buffers 4]\n"
    "  --grid         voxels per axis (power of two)\n"
    "  --block        voxels per block axis (power of two)\n"
    "  --list-budget  triangle list memory per batch, bytes with optional K/M/G suffix\n"
    "  --buffers      compact buffers in flight between GPU and disk\n";

uint64_t parseSize(const std::string& s) {
    // stoull alone would accept leading whitespace and wrap "-1" around.
    if (s.empty() || !std::isdigit(uint8_t(s[0]))) throw std::invalid_argument("bad size: " + s);
    size_t end = 0;
    uint64_t value = std::stoull(s, &end);
    std::string suffix = s.substr(end);
    int shift = 0;
    if (suffix == "K" || suffix == "k") shift = 10;
    else if (suffix == "M" || suffix == "m") shift = 20;
    else if (suffix == "G" || suffix == "g") shift = 30;
    else if (!suffix.empty()) throw std::invalid_argument("bad size: " + s);
    if (value > UINT64_MAX >> shift) throw std::out_of_range("size too large: " + s);
    return value << shift;
}

uint32_t parseU32(const std::string& s) {
    uint64_t value = parseSize(s);
    if (value > UINT32_MAX) throw std::out_of_range("value too large: " + s);
    return uint32_t(value);
}

// Loads all triangles of the OBJ, scaled uniformly into [0, grid)^3 with the bounding box minimum at 0.
// Zero-area triangles are dropped.
Mesh loadObj(const std::string& path, uint32_t grid) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = false;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, config)) throw std::runtime_error(std::format("cannot load {}: {}", path, reader.Error()));

    const std::vector<float>& v = reader.GetAttrib().vertices;
    if (v.empty()) throw std::runtime_error(path + " has no vertices");
    float lo[3] = {v[0], v[1], v[2]}, hi[3] = {v[0], v[1], v[2]};
    for (size_t i = 0; i < v.size(); i += 3)
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], v[i + a]);
            hi[a] = std::max(hi[a], v[i + a]);
        }
    double extent = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    double scale = extent > 0 ? grid / extent : 1.0;
    const float top = std::nextafter(float(grid), 0.0f);

    Mesh mesh;
    mesh.xyz.resize(v.size());
    for (size_t i = 0; i < v.size(); i += 3)
        for (int a = 0; a < 3; ++a) mesh.xyz[i + a] = std::min(float((v[i + a] - lo[a]) * scale), top);

    auto zeroArea = [&](uint32_t i0, uint32_t i1, uint32_t i2) {
        const float *a = &mesh.xyz[i0 * 3], *b = &mesh.xyz[i1 * 3], *c = &mesh.xyz[i2 * 3];
        float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        return u[1] * w[2] - u[2] * w[1] == 0 && u[2] * w[0] - u[0] * w[2] == 0 && u[0] * w[1] - u[1] * w[0] == 0;
    };
    for (const tinyobj::shape_t& shape : reader.GetShapes()) {
        const auto& idx = shape.mesh.indices; // triangulated: 3 per face
        for (size_t i = 0; i + 2 < idx.size(); i += 3) {
            uint32_t i0 = idx[i].vertex_index, i1 = idx[i + 1].vertex_index, i2 = idx[i + 2].vertex_index;
            if (!zeroArea(i0, i1, i2)) mesh.indices.insert(mesh.indices.end(), {i0, i1, i2});
        }
    }
    return mesh;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> positional;
        Params params;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.starts_with("--")) {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                std::string value = argv[++i];
                if (arg == "--grid") params.grid = parseU32(value);
                else if (arg == "--block") params.block = parseU32(value);
                else if (arg == "--list-budget") params.listBudget = parseSize(value);
                else if (arg == "--buffers") params.buffers = parseU32(value);
                else throw std::invalid_argument("unknown option " + arg);
            } else {
                positional.push_back(arg);
            }
        }
        if (positional.size() != 2 || params.grid == 0) {
            std::fputs(kUsage, stderr);
            return 2;
        }

        std::fprintf(stderr, "loading %s (%.1f MiB)...\n", positional[0].c_str(),
                     double(std::filesystem::file_size(positional[0])) / (1 << 20));
        auto t0 = std::chrono::steady_clock::now();
        Mesh mesh = loadObj(positional[0], params.grid);
        auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "loaded %zu triangles in %.2f s\n", mesh.indices.size() / 3, std::chrono::duration<double>(t1 - t0).count());
        Stats stats = voxelize(mesh, params, positional[1]);
        auto t2 = std::chrono::steady_clock::now();

        auto seconds = [](auto d) { return std::chrono::duration<double>(d).count(); };
        double written = double(stats.bricks * kRecordBytes);
        std::printf("%zu triangles, %llu batches, %llu blocks, %llu bricks (%.1f MiB)\n", mesh.indices.size() / 3,
                    (unsigned long long)stats.batches, (unsigned long long)stats.blocks, (unsigned long long)stats.bricks,
                    written / (1 << 20));
        std::printf("load %.2f s, voxelize %.2f s (%.1f MiB/s)\n", seconds(t1 - t0), seconds(t2 - t1),
                    written / (1 << 20) / seconds(t2 - t1));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
