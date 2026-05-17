# Voxelizer

A high-performance, multithreaded command-line tool that converts `.obj` mesh files into a compact binary voxel representation. Voxels are stored as **sorted, deduplicated 64-bit Morton codes**, making the output directly suitable for spatial data structures such as Sparse Voxel Octrees (SVO) and DAGs.

---

## Features

- **Multithreaded** — parallelizes voxelization across all available CPU cores
- **Out-of-core processing** — writes voxel data to temporary chunk files when RAM is full, then performs a k-way merge; output size is limited only by available disk space
- **Large mesh support** — automatically subdivides triangles whose AABB spans too many voxels, keeping per-thread memory usage bounded
- **Sorted & deduplicated output** — Morton-order sort and full deduplication happen at every stage (per chunk and during merge), so the final file is clean and compact
- **Configurable** — resolution, RAM budget, and subdivision threshold are all adjustable via CLI flags

---

## Output Format

The `.bin` output file is a raw binary array of `uint64_t` Morton codes:

- **No header** — the file starts immediately with the first code
- **Sorted ascending** — codes are in Z-order (Morton order)
- **Deduplicated** — each voxel appears exactly once
- **Element size** — 8 bytes per voxel
- **Coordinate range** — each axis is clamped to `[0, resolution - 1]`

To read the file, treat it as a flat array of `uint64_t`. The number of voxels is `file_size / 8`.

---

## Algorithm Overview

```
parse_args()
    └─ Validates paths, resolves temp directory

load_obj()
    └─ Parses the .obj file via tinyobjloader
    └─ Flattens all shapes/faces into a Triangle list
    └─ Computes the model's axis-aligned bounding box (AABB)

subdivide_triangles()
    └─ For each triangle, estimates its AABB footprint in voxels
    └─ If footprint > --max-voxels-tri, splits along the longest edge
    └─ Repeats until all triangles are within the threshold

voxelize()  [multithreaded]
    └─ Each worker thread claims triangles atomically
    └─ For each triangle, computes the candidate voxel AABB
    └─ Tests each candidate voxel using SAT (Separating Axis Theorem)
    └─ On hit: encodes voxel (x, y, z) as a Morton code
    └─ When local buffer reaches --max-ram limit: sorts, deduplicates,
       and flushes to a temp chunk file on disk
    └─ After all workers finish: k-way merge of all chunk files
       (via a min-heap priority queue), with final deduplication
    └─ Writes the merged result to the output .bin file
```

### Intersection Test

Each candidate voxel is tested against its triangle using the **Separating Axis Theorem (SAT)** — the same approach as Möller's triangle-box overlap test. This covers:
1. AABB overlap check
2. Nine edge-cross-product axes
3. Triangle plane axis

### Morton Encoding

Voxel coordinates are encoded into 64-bit Morton codes using bit-interleaving (`split_by_3`), supporting up to 21 bits per axis (max resolution ~2,097,152 per side). At the default resolution of 1024, this is well within range.

---

## Limitations

### Input file size

The `.obj` parser (tinyobjloader) buffers the entire file into memory before parsing. The internal limit is:

```
~2024 MB (~1.98 GB)
```

This can be increased by defining `TINYOBJLOADER_STREAM_READER_MAX_BYTES` to a larger value before including `tiny_obj_loader.h`.

### Output file size / number of chunks

Output size is limited by available disk space. The number of simultaneously open chunk files during the merge phase is bounded by the OS limit on open file descriptors. On most Linux systems the default soft limit is 1024 (adjustable via `ulimit -n`). On Windows the practical limit is in the thousands.

### Non-triangulated faces

tinyobjloader automatically triangulates quads and n-gons before returning results (`triangulate = true` by default, using simple fan triangulation). All face types in the `.obj` are therefore supported.

---

## Building

**Requirements:**
- CMake ≥ 3.8
- A C++23-capable compiler (MSVC 2022, GCC 13+, Clang 16+)
- Internet access on first build (GLM is fetched via `FetchContent`)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The binary is placed in `build/bin/voxelizer` (or `build\bin\Release\voxelizer.exe` on Windows).

---

## Usage

```
voxelizer -i <input.obj> -o <output.bin> [options]
```

### Required arguments

| Flag | Description |
|---|---|
| `-i, --input` | Path to the input `.obj` file |
| `-o, --output` | Path for the output `.bin` file (`.bin` extension added automatically if missing) |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `-r, --resolution` | `1024` | Voxel grid resolution (applies to the longest model axis) |
| `-m, --max-ram` | `1000000000` | Maximum total voxels buffered in RAM across all threads before flushing to disk |
| `--max-voxels-tri` | `100000` | Maximum voxels in a triangle's AABB before it gets subdivided |
| `-t, --temp` | `<output_dir>/temp` | Directory for temporary chunk files (must already exist if specified) |

### Examples

**Basic voxelization at default resolution (1024):**
```bash
voxelizer -i model.obj -o model.bin
```

**High-resolution voxelization (4096³):**
```bash
voxelizer -i model.obj -o model_4096.bin -r 4096
```

**Low RAM budget (useful on memory-constrained machines):**
```bash
voxelizer -i model.obj -o model.bin -m 100000000
```

**Custom temp directory:**
```bash
voxelizer -i model.obj -o model.bin -t D:\scratch
```

---

## Dependencies

| Library | Version | How |
|---|---|---|
| [GLM](https://github.com/g-truc/glm) | 0.9.9.8 | CMake `FetchContent` |
| [CLI11](https://github.com/CLIUtils/CLI11) | bundled | `third_party/CLI11` |
| [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) | bundled | `third_party/tinyobjloader` |

---

## Project Structure

```
voxelizer/
├── main.cpp                 # Entry point and pipeline stages
├── CMakeLists.txt
├── geometry/
│   ├── aabb.hpp             # Axis-aligned bounding box
│   └── intersection.hpp     # SAT voxel-triangle intersection test
├── math/
│   └── morton.hpp           # 64-bit Morton code encoding
└── third_party/
    ├── CLI11/
    └── tinyobjloader/
```
