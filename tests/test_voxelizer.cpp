// End-to-end check: voxelizes one scene with several block sizes, list budgets and buffer counts.
// Every configuration must produce the same file (binning never loses a triangle, keys are absolute,
// blocks come out in Morton order), and the file must be a strictly sorted list of non-empty bricks.
// usage: test_voxelizer <voxelizer executable> <scratch dir>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        std::exit(1);
    }
}

// UV sphere, tilted torus and one large slanted triangle crossing many blocks.
void writeScene(const fs::path& path) {
    std::ofstream obj(path);
    int base = 1;
    auto grid = [&](int nu, int nv, auto position) {
        for (int i = 0; i <= nu; ++i)
            for (int j = 0; j <= nv; ++j) {
                auto [x, y, z] = position(2 * std::numbers::pi * i / nu, std::numbers::pi * j / nv);
                obj << "v " << x << ' ' << y << ' ' << z << '\n';
            }
        for (int i = 0; i < nu; ++i)
            for (int j = 0; j < nv; ++j) {
                int a = base + i * (nv + 1) + j, b = a + nv + 1;
                obj << "f " << a << ' ' << b << ' ' << b + 1 << ' ' << a + 1 << '\n';
            }
        base += (nu + 1) * (nv + 1);
    };
    grid(64, 32, [](double u, double v) {
        return std::tuple{std::cos(u) * std::sin(v), std::sin(u) * std::sin(v), std::cos(v)};
    });
    grid(96, 48, [](double u, double v) {
        double r = 0.8 + 0.25 * std::cos(2 * v), x = r * std::cos(u), y = r * std::sin(u), z = 0.25 * std::sin(2 * v);
        return std::tuple{1.5 + x, 0.3 + 0.8 * y - 0.6 * z, 0.2 + 0.6 * y + 0.8 * z};
    });
    obj << "v -1.2 -1.0 -0.8\nv 2.4 -0.6 0.9\nv 0.3 1.2 1.1\n";
    obj << "f " << base << ' ' << base + 1 << ' ' << base + 2 << '\n';
}

std::vector<uint64_t> run(const std::string& exe, const fs::path& dir, const std::string& args) {
    fs::path out = dir / "out.bin";
    std::string cmd = "\"" + exe + "\" \"" + (dir / "scene.obj").string() + "\" \"" + out.string() + "\" " + args;
#ifdef _WIN32
    cmd = "\"" + cmd + "\""; // cmd.exe strips the outer quotes
#endif
    std::printf("%s\n", args.c_str());
    std::fflush(stdout);
    check(std::system(cmd.c_str()) == 0, "voxelizer failed: " + args);
    std::vector<uint64_t> data(fs::file_size(out) / 8);
    std::ifstream(out, std::ios::binary).read(reinterpret_cast<char*>(data.data()), data.size() * 8);
    return data;
}

int main(int argc, char** argv) {
    check(argc == 3, "usage: test_voxelizer <voxelizer> <scratch dir>");
    fs::path dir = argv[2];
    fs::create_directories(dir);
    writeScene(dir / "scene.obj");

    const uint32_t grid = 128;
    auto ref = run(argv[1], dir, "--grid 128 --block 128");
    check(!ref.empty() && ref.size() % 2 == 0, "reference output empty or odd");
    for (size_t i = 0; i < ref.size(); i += 2) {
        check(ref[i + 1] != 0, "empty brick in output");
        check(ref[i] < uint64_t(grid / 4) * (grid / 4) * (grid / 4), "key outside the grid");
        check(i == 0 || ref[i] > ref[i - 2], "keys not strictly increasing");
    }
    std::printf("reference: %zu bricks\n", ref.size() / 2);

    for (const char* args : {"--grid 128 --block 64 --buffers 1", "--grid 128 --block 32 --list-budget 64K --buffers 2",
                             "--grid 128 --block 16 --list-budget 16K", "--grid 128 --block 4 --list-budget 64K --buffers 3"})
        check(run(argv[1], dir, args) == ref, std::string("output differs from reference: ") + args);

    std::printf("OK\n");
    return 0;
}
