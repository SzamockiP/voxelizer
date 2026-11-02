# Voxelizer

Voxelizer is a command-line tool. It converts an `.obj` mesh to a sparse voxel file on the GPU with Vulkan.

The tool rasterizes the triangles with conservative rasterization. The result is a surface voxelization. The tool stores only the occupied voxels. For this reason, the output size depends on the surface area of the model and not on the grid volume. The grid can have up to 2^21 voxels per axis. See [Limits](#limits).

The branch `cpu` has the older CPU version of this tool.

## Build

You need these items:

- CMake 3.25 or later
- A C++23 compiler (MSVC 2022, GCC 13 or later, Clang 17 or later)
- The [Vulkan SDK](https://vulkan.lunarg.com/) (headers, loader, and `glslc`)
- Internet access for the first configure step. CMake downloads tinyobjloader and VulkanMemoryAllocator.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## GPU requirements

The GPU must support these items:

- Vulkan 1.3
- `VK_EXT_conservative_rasterization` with an overestimation of at least 0.25 pixels
- 64-bit shader integers and 64-bit buffer atomics
- Buffer device address, timeline semaphores, and dynamic rendering
- Subgroup ballot and arithmetic operations in compute shaders
- Framebuffer and viewport limits that are large enough for the block size

The tool selects a discrete GPU when one is available. If no GPU meets the requirements, the tool shows what each GPU does not have.

Resizable BAR is recommended. The tool puts the vertex and index buffers in device-local memory that the CPU can also access. Without Resizable BAR, this memory is usually only 256 MiB, so large meshes fail to load.

## Usage

```
voxelizer <model.obj> <out.bin> --grid N [--block 1024] [--list-budget 1G] [--buffers 4]
```

| Option | Default | Description |
|---|---|---|
| `--grid` | required | Voxels per axis. The value must be a power of two. It must be at least 4 and at most 1024 times `--block`. The tool scales the model so that its longest side fits the grid. |
| `--block` | `1024` | Voxels per axis in one block. The value must be a power of two from 4 to 2048, and not more than `--grid`. |
| `--list-budget` | `1G` | Memory for the triangle list of one batch, in bytes. You can add the suffix `K`, `M`, or `G`. The triangle list of the largest block must fit in this memory. |
| `--buffers` | `4` | Number of compact buffers that the GPU and the disk writer share. |

Examples:

```bash
voxelizer model.obj model.bin --grid 4096
voxelizer model.obj model.bin --grid 16384 --block 512 --list-budget 512M
```

## How it works

1. The tool loads the OBJ file and triangulates it. It scales the model into `[0, grid)^3` and removes triangles that have zero area.
2. The tool counts the triangles that touch each block. It then divides the blocks into batches that fit in `--list-budget`. For each batch, the tool draws the triangles a second time to fill the triangle list.
3. The tool rasterizes each non-empty block at full resolution. The result is a dense array of bricks. Each brick has 4x4x4 voxels.
4. Compute shaders find the non-empty bricks with a prefix sum. They write these bricks directly to host memory.
5. A writer thread saves each finished block to the disk while the GPU works on the next block. The tool processes the blocks in Morton order, so the file is sorted.

## Limits

The tool divides the grid into blocks. Two rules limit the sizes:

- `--block` is at most 2048. The tool counts the bricks of a block in groups of 256 bricks. It then adds the groups in runs of 1024 groups. One workgroup of 1024 threads adds the runs. A block of 2048 has 512 runs. A block of 4096 has 4096 runs, which is too many.
- `--grid / --block` is at most 1024 blocks per axis. The Morton index of a block has 10 bits per axis.

For this reason, the maximum grid is 1024 times the block size. With the default block of 1024, the maximum grid is 2^20. With a block of 2048, it is 2^21.

Memory is usually the tighter limit. The tool needs these amounts:

| Item | Where | Size |
|---|---|---|
| Block counters | GPU and RAM (one copy each) | `(grid / block)^3 * 4` bytes |
| Dense block | GPU | `(block / 4)^3 * 8` bytes |
| Compact buffers | RAM | `--buffers * (block / 4)^3 * 16` bytes |
| Triangle list | GPU | `--list-budget` |
| Mesh | GPU | 12 bytes per vertex and 12 bytes per triangle |

Typical values:

| Blocks per axis | Block counters (each copy) |
|---|---|
| 64 | 1 MiB |
| 128 | 8 MiB |
| 256 | 64 MiB |
| 512 | 512 MiB |
| 1024 | 4 GiB |

| `--block` | Dense block | One compact buffer |
|---|---|---|
| 256 | 2 MiB | 4 MiB |
| 512 | 16 MiB | 32 MiB |
| 1024 | 128 MiB | 256 MiB |
| 2048 | 1 GiB | 2 GiB |

Example: `--grid 16384 --block 1024` has 16 blocks per axis. The block counters need 16 KiB for each copy. The dense block needs 128 MiB of GPU memory. Four compact buffers need 1 GiB of RAM.

The tool stops with an error when one block needs more triangle list entries than `--list-budget` holds. In this case, increase `--list-budget` or decrease `--block`.

## Output format

The output is a flat binary file with no header. It uses little-endian byte order. Each record has 16 bytes:

| Field | Type | Description |
|---|---|---|
| key | `uint64` | Morton code of the brick position `(x/4, y/4, z/4)`. The x coordinate uses the lowest bit of each group of three bits. |
| mask | `uint64` | The occupied voxels of the brick. Bit `i` is `(x & 3) \| (y & 3) << 2 \| (z & 3) << 4`. |

The records are in strict ascending order of key. No brick is empty. To get the number of records, divide the file size by 16.

## Tests and tools

To run the test, use this command:

```bash
ctest --test-dir build -C Release --output-on-failure
```

The test needs a Vulkan GPU. It generates a scene and voxelizes it with different block sizes, list budgets, and buffer counts. It then makes sure that all runs give the same sorted output.

The script `tools/views.py` needs `numpy` and `Pillow`. It casts rays along the six axis directions and compares the voxel file with the OBJ file. It writes images that show the voxels, the mesh, and the differences side by side.

```bash
python tools/views.py model.obj model.bin 4096 views/
```
