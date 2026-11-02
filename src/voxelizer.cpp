#include "voxelizer.hpp"

#include "queue.hpp"
#include "vk.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

// Compact buffer header: record count and padding (see Records in compact.glsl).
constexpr uint64_t kHeaderBytes = 16;

uint32_t compact1by2(uint32_t x) {
    x &= 0x09249249u;
    x = (x ^ (x >> 2)) & 0x030C30C3u;
    x = (x ^ (x >> 4)) & 0x0300F00Fu;
    x = (x ^ (x >> 8)) & 0xFF0000FFu;
    x = (x ^ (x >> 16)) & 0x000003FFu;
    return x;
}

void validate(const Mesh& mesh, const Params& p) {
    auto fail = [](const std::string& message) { throw std::invalid_argument(message); };
    if (!std::has_single_bit(p.grid) || p.grid < 4) fail(std::format("grid must be a power of two of at least 4, got {}", p.grid));
    // 2048: the scan of the run totals is one workgroup of 1024 threads (see compact_scan_totals.comp).
    if (!std::has_single_bit(p.block) || p.block < 4 || p.block > 2048 || p.block > p.grid)
        fail(std::format("block must be a power of two in [4, min(2048, grid)], got {}", p.block));
    // 1024: block Morton indices have 10 bits per axis. Together with the block limit this caps grid at 2^21.
    if (p.grid / p.block > 1024)
        fail(std::format("grid/block = {} blocks per axis, at most 1024 are supported (grid <= 1024 * block)", p.grid / p.block));
    if (p.buffers < 1) fail("need at least one compact buffer");
    // List entries are addressed through gl_VertexIndex = entry * 3.
    if (p.listBudget < 4 || p.listBudget / 4 * 3 > UINT32_MAX)
        fail(std::format("list budget must be in [4 B, {} B]", uint64_t(UINT32_MAX) / 3 * 4));
    if (mesh.indices.empty()) fail("model has no triangles");
    if (mesh.indices.size() > UINT32_MAX) fail("model has too many triangles");
}

struct Batch {
    uint32_t first, last; // Morton range of blocks
    uint32_t total;       // list entries
};

// Prefix sum of the per-block entry counts in place, split into batches of at most `maxEntries`. Each
// counter becomes the end of its block's range, relative to the start of its batch.
std::vector<Batch> planBatches(uint32_t* counters, uint32_t blockCount, uint64_t maxEntries) {
    std::vector<Batch> batches;
    uint64_t sum = 0;
    uint32_t first = 0;
    for (uint32_t b = 0; b < blockCount; ++b) {
        uint32_t c = counters[b];
        if (c > maxEntries)
            throw std::runtime_error(std::format("block {} needs {} list entries, the list budget holds {}", b, c, maxEntries));
        if (sum + c > maxEntries) {
            batches.push_back({first, b - 1, uint32_t(sum)});
            first = b;
            sum = 0;
        }
        sum += c;
        counters[b] = uint32_t(sum);
    }
    batches.push_back({first, blockCount - 1, uint32_t(sum)});
    return batches;
}

void copyCounters(VkCommandBuffer cmd, const Buffer& from, const Buffer& to, uint32_t firstBlock, uint32_t lastBlock) {
    VkBufferCopy region{uint64_t(firstBlock) * 4, uint64_t(firstBlock) * 4, uint64_t(lastBlock - firstBlock + 1) * 4};
    vkCmdCopyBuffer(cmd, from.buffer, to.buffer, 1, &region);
}

// Rasterizes one block from list entries [start, end) and compacts it into the compact buffer of `compactPC`.
void recordBlock(Gpu& gpu, VkCommandBuffer cmd, const DrawPC& blockPC, const CompactPC& compactPC, uint32_t start, uint32_t end) {
    const uint32_t runCount = (compactPC.groupCount + 1023) / 1024;
    gpu.begin(cmd);
    gpu.barrier(cmd); // previous block's compaction (and the fill pass) before this block
    gpu.draw(cmd, gpu.voxelPipeline, blockPC, (end - start) * 3, start * 3); // also counts bricks per group
    gpu.barrier(cmd);
    gpu.dispatch(cmd, gpu.scanGroupsPipeline, compactPC, runCount);
    gpu.barrier(cmd);
    gpu.dispatch(cmd, gpu.scanTotalsPipeline, compactPC, 1);
    gpu.barrier(cmd);
    gpu.dispatch(cmd, gpu.writePipeline, compactPC, compactPC.groupCount);
    gpu.end(cmd);
}

} // namespace

Stats voxelize(const Mesh& mesh, const Params& p, const std::filesystem::path& outPath) {
    validate(mesh, p);
    const uint32_t blocksPerAxis = p.grid / p.block;
    const uint32_t blockCount = blocksPerAxis * blocksPerAxis * blocksPerAxis;
    const uint32_t bricksPerAxis = p.block / 4;
    const uint32_t brickCount = bricksPerAxis * bricksPerAxis * bricksPerAxis;
    const uint32_t groupCount = (brickCount + 255) / 256; // compaction groups of 256 bricks
    const uint32_t runCount = (groupCount + 1023) / 1024;  // first-level scan workgroups
    const uint32_t brickBits = 3 * std::countr_zero(bricksPerAxis);
    const uint32_t triangleCount = uint32_t(mesh.indices.size() / 3);
    const uint64_t maxEntries = p.listBudget / 4;

    Gpu gpu(std::max(blocksPerAxis, p.block));
    std::fprintf(stderr, "initialized GPU, %u x %u x %u blocks of %u^3 voxels\n", blocksPerAxis, blocksPerAxis, blocksPerAxis, p.block);
    Buffer verts = gpu.createBuffer("vertex buffer", mesh.xyz.size() * 4, Memory::Bar);
    Buffer indices = gpu.createBuffer("index buffer", mesh.indices.size() * 4, Memory::Bar);
    std::memcpy(verts.mapped, mesh.xyz.data(), mesh.xyz.size() * 4);
    std::memcpy(indices.mapped, mesh.indices.data(), mesh.indices.size() * 4);
    Buffer counters = gpu.createBuffer("block counters", uint64_t(blockCount) * 4, Memory::Device);
    // The CPU works on a copy in cached RAM: reading BAR memory directly runs at a few MB/s.
    Buffer mirror = gpu.createBuffer("block counters mirror", uint64_t(blockCount) * 4, Memory::HostCached);
    Buffer list = gpu.createBuffer("triangle list", p.listBudget, Memory::Device);
    Buffer dense = gpu.createBuffer("dense block", uint64_t(brickCount) * 8, Memory::Device);
    Buffer groupCounts = gpu.createBuffer("compaction group counts", uint64_t(groupCount) * 4, Memory::Device);
    Buffer groupOffsets = gpu.createBuffer("compaction group offsets", uint64_t(groupCount) * 4, Memory::Device);
    Buffer runTotals = gpu.createBuffer("compaction run totals", uint64_t(runCount) * 4, Memory::Device);
    // Per slot: compaction writes the records straight into host memory, the writer reads them from there.
    std::vector<Buffer> compact;
    std::vector<VkCommandBuffer> slotCmds;
    for (uint32_t i = 0; i < p.buffers; ++i) {
        compact.push_back(gpu.createBuffer("compact buffer", kHeaderBytes + uint64_t(brickCount) * kRecordBytes, Memory::HostCached));
        slotCmds.push_back(gpu.createCommandBuffer());
    }
    VkCommandBuffer passCmd = gpu.createCommandBuffer();
    auto* hostCounters = static_cast<uint32_t*>(mirror.mapped);

    DrawPC binPC{.verts = verts.address, .indices = indices.address, .list = list.address, .counters = counters.address,
                 .pixelSize = float(p.block), .size = blocksPerAxis, .mode = MODE_COUNT};

    std::fprintf(stderr, "binning %u triangles...\n", triangleCount);
    // Binning, counting pass. Also clears the dense block and group counts once; compaction keeps them clear.
    gpu.begin(passCmd);
    vkCmdFillBuffer(passCmd, counters.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(passCmd, dense.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(passCmd, groupCounts.buffer, 0, VK_WHOLE_SIZE, 0);
    gpu.barrier(passCmd);
    gpu.draw(passCmd, gpu.binPipeline, binPC, triangleCount * 3, 0);
    gpu.barrier(passCmd);
    copyCounters(passCmd, counters, mirror, 0, blockCount - 1);
    gpu.barrier(passCmd);
    gpu.end(passCmd);
    gpu.wait(gpu.submit(passCmd));

    const uint64_t occupiedBlocks = std::count_if(hostCounters, hostCounters + blockCount, [](uint32_t c) { return c != 0; });
    const std::vector<Batch> batches = planBatches(hostCounters, blockCount, maxEntries);
    std::fprintf(stderr, "voxelizing %llu occupied blocks in %zu batches\n", (unsigned long long)occupiedBlocks, batches.size());

    Stats stats;
    std::ofstream file(outPath, std::ios::binary);
    if (!file) throw std::runtime_error(std::format("cannot open {}", outPath.string()));
    file.rdbuf()->pubsetbuf(nullptr, 0);

    // Writer: takes filled compact buffers in submission (= Morton) order, waits for the GPU, writes the
    // records and hands the buffer back.
    Queue<uint32_t> freeSlots;
    Queue<std::pair<uint32_t, uint64_t>> fullSlots; // slot, timeline value of its block
    for (uint32_t i = 0; i < p.buffers; ++i) freeSlots.push(i);
    std::exception_ptr writerError, producerError;

    // Progress is measured on the writer side (blocks on disk), printed at most once a second.
    using Clock = std::chrono::steady_clock;
    uint64_t doneBlocks = 0, lastBlocks = 0, lastBricks = 0;
    const auto writeStart = Clock::now();
    auto lastReport = writeStart;
    auto report = [&](bool final) {
        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - lastReport).count();
        if (!final && dt < 1.0) return;
        if (final) dt = std::chrono::duration<double>(now - writeStart).count(), lastBlocks = lastBricks = 0; // overall average
        double mib = double(stats.bricks * kRecordBytes) / (1 << 20);
        double rateMib = double(stats.bricks - lastBricks) * kRecordBytes / (1 << 20) / dt;
        std::fprintf(stderr, "  %llu / %llu blocks, %.1f MiB written, %.1f MiB/s, %.1f blocks/s   %s",
                     (unsigned long long)doneBlocks, (unsigned long long)occupiedBlocks, mib, rateMib,
                     double(doneBlocks - lastBlocks) / dt, final ? "\n" : "");
        lastReport = now, lastBlocks = doneBlocks, lastBricks = stats.bricks;
    };
    std::jthread writer([&] {
        try {
            while (auto job = fullSlots.pop()) {
                auto [slot, value] = *job;
                gpu.wait(value);
                const auto* data = static_cast<const char*>(compact[slot].mapped);
                uint64_t count;
                std::memcpy(&count, data, 8);
                if (!file.write(data + kHeaderBytes, std::streamsize(count * kRecordBytes)))
                    throw std::runtime_error(std::format("writing {} failed", outPath.string()));
                stats.bricks += count;
                ++doneBlocks;
                report(false);
                freeSlots.push(slot);
            }
        } catch (...) {
            writerError = std::current_exception();
            freeSlots.close();
        }
    });

    // Producer: per batch fill the list, then queue every non-empty block of the batch.
    try {
        for (const Batch& batch : batches) {
            if (batch.total == 0) continue;
            ++stats.batches;
            gpu.wait(gpu.lastSubmitted()); // previous batch's blocks still read the list

            binPC.mode = MODE_FILL;
            binPC.first = batch.first;
            binPC.last = batch.last;
            gpu.begin(passCmd);
            gpu.barrier(passCmd);
            copyCounters(passCmd, mirror, counters, batch.first, batch.last); // range ends
            gpu.barrier(passCmd);
            gpu.draw(passCmd, gpu.binPipeline, binPC, triangleCount * 3, 0);
            gpu.barrier(passCmd);
            copyCounters(passCmd, counters, mirror, batch.first, batch.last); // range starts
            gpu.barrier(passCmd);
            gpu.end(passCmd);
            gpu.wait(gpu.submit(passCmd)); // counters now hold range starts

            for (uint32_t b = batch.first; b <= batch.last; ++b) {
                uint32_t start = hostCounters[b];
                uint32_t end = b < batch.last ? hostCounters[b + 1] : batch.total;
                if (start == end) continue;
                auto slot = freeSlots.pop();
                if (!slot) throw std::runtime_error("writer stopped");

                uint32_t bx = compact1by2(b), by = compact1by2(b >> 1), bz = compact1by2(b >> 2);
                const DrawPC blockPC{.verts = verts.address, .indices = indices.address, .list = list.address,
                                     .origin = {float(bx * p.block), float(by * p.block), float(bz * p.block)},
                                     .pixelSize = 1.0f, .size = p.block, .mode = MODE_BLOCK,
                                     .dense = dense.address, .groupCounts = groupCounts.address};
                const CompactPC compactPC{.dense = dense.address, .counts = groupCounts.address,
                                          .offsets = groupOffsets.address, .totals = runTotals.address,
                                          .records = compact[*slot].address, .keyPrefix = uint64_t(b) << brickBits,
                                          .brickCount = brickCount, .groupCount = groupCount};
                recordBlock(gpu, slotCmds[*slot], blockPC, compactPC, start, end);
                fullSlots.push({*slot, gpu.submit(slotCmds[*slot])});
                ++stats.blocks;
            }
        }
    } catch (...) {
        producerError = std::current_exception();
    }
    fullSlots.close();
    writer.join();
    if (writerError) std::rethrow_exception(writerError);
    if (producerError) std::rethrow_exception(producerError);
    report(true);

    file.close();
    if (!file) throw std::runtime_error(std::format("writing {} failed", outPath.string()));
    return stats;
}
