"""Compares a voxelization with its OBJ by orthographic "ray casting" along all 6 axis directions.

For every view: first hit depth per pixel from the voxels, and from dense random points on the triangles
(scaled exactly like main.cpp). Writes side-by-side shaded images (voxels | mesh | errors) and prints stats.
usage: views.py <model.obj> <voxels.bin> <grid> <out dir>
"""
import os, sys, time
import numpy as np
from PIL import Image

obj, binf, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
t0 = time.perf_counter()

# --- mesh, as main.cpp: all vertices for the bbox, uniform scale grid/extent, clamp below grid
lines = open(obj, 'rb').read().split(b'\n')
V = np.array(b' '.join(l[2:] for l in lines if l.startswith(b'v ')).split(), np.float32).reshape(-1, 3)
F = []
for l in lines:
    if l.startswith(b'f '):
        ids = [int(t.split(b'/')[0]) for t in l.split()[1:]]
        ids = [i - 1 if i > 0 else len(V) + i for i in ids]
        F.extend((ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1))
del lines
F = np.array(F, np.int64)
lo, hi = V.min(0), V.max(0)
scale = N / float((hi - lo).max())
top = np.nextafter(np.float32(N), np.float32(0))
P = np.minimum(((V.astype(np.float64) - lo) * scale).astype(np.float32), top)
T = P[F].astype(np.float64)
n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
area = 0.5 * np.linalg.norm(n, axis=1)
T, area = T[area > 0], area[area > 0]
print(f"mesh: {len(T)} triangles, surface {area.sum():.3g} voxel^2, loaded in {time.perf_counter() - t0:.1f} s")

# --- voxels
d = np.fromfile(binf, np.uint64).reshape(-1, 2)
def unspread(x):
    x = x & np.uint64(0x1249249249249249)
    for s, m in [(2, 0x10c30c30c30c30c3), (4, 0x100f00f00f00f00f), (8, 0x1f0000ff0000ff), (16, 0x1f00000000ffff), (32, 0x1fffff)]:
        x = (x ^ (x >> np.uint64(s))) & np.uint64(m)
    return x.astype(np.int64)
B = np.stack([unspread(d[:, 0] >> np.uint64(a)) for a in range(3)], 1) * 4
vox = []
for bit in range(64):
    sel = ((d[:, 1] >> np.uint64(bit)) & np.uint64(1)).astype(bool)
    vox.append(B[sel] + [bit & 3, (bit >> 2) & 3, bit >> 4])
vox = np.concatenate(vox)
print(f"voxels: {len(vox)}")

# --- first-hit depth maps; view (a, side): rays along axis a, from the min side (0) or the max side (1)
FAR = 1 << 30
def depth_maps(cells, maps):
    for a in range(3):
        u, v = [(a + 1) % 3, (a + 2) % 3]
        idx = cells[:, u] * N + cells[:, v]
        np.minimum.at(maps[a][0], idx, cells[:, a])
        np.minimum.at(maps[a][1], idx, N - 1 - cells[:, a])
new = lambda: [[np.full(N * N, FAR, np.int64) for _ in range(2)] for _ in range(3)]
vmap, mmap = new(), new()
depth_maps(vox, vmap)

rng = np.random.default_rng(1)
# ~4 samples per voxel^2, plus enough that long slivers get one every ~1/3 voxel along their length
edge = np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2).max(1)
count = (area * 4 + edge * 3).astype(np.int64) + 3
samples = 0
for s in range(0, len(T), 200_000):
    t, c = T[s:s + 200_000], count[s:s + 200_000]
    tri = np.repeat(np.arange(len(t)), c)
    r1, r2 = np.sqrt(rng.random(len(tri))), rng.random(len(tri))
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    pts = np.einsum('pk,pkd->pd', w, t[tri])
    depth_maps(np.floor(pts).astype(np.int64), mmap)
    samples += len(tri)
print(f"mesh samples: {samples}")

def nb(m, f):  # f over the 3x3 neighbourhood
    p = np.pad(m, 1, constant_values=FAR if f is np.minimum else -1)
    r = m.copy()
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            r = f(r, p[1 + i:1 + i + N, 1 + j:1 + j + N])
    return r

def shade(depth):
    hit = depth < FAR
    z = np.where(hit, depth, 0).astype(np.float64)
    gx, gy = np.gradient(z)
    light = np.clip(0.55 + 0.35 * (gx - gy) / np.sqrt(1 + gx * gx + gy * gy), 0, 1)
    fade = 1 - 0.6 * z / N
    img = np.where(hit, light * fade, 0.12)
    return (np.stack([img] * 3, -1) * 255).astype(np.uint8)

names = 'xyz'
total_err = 0
for a in range(3):
    for side in range(2):
        vd, md = vmap[a][side].reshape(N, N), mmap[a][side].reshape(N, N)
        vh, mh = vd < FAR, md < FAR
        both = vh & mh
        diff = (vd - md)[both]
        # holes: the ray hits the mesh but passes through the voxels (no voxel within 2 behind the hit, 3x3)
        hole = mh & (nb(vd, np.minimum) > md + 2)
        # strays: the ray stops at voxels with no mesh nearby (more than 2 in front of every mesh hit, 3x3)
        stray = vh & (vd < nb(np.where(mh, md, FAR), np.minimum) - 2)
        err = hole | stray
        total_err += err.sum()
        print(f"view {'+-'[side]}{names[a]}: pixels voxel {vh.sum()}, mesh {mh.sum()}, "
              f"|depth diff| <=1: {np.mean(np.abs(diff) <= 1) * 100:.3f}%, <=2: {np.mean(np.abs(diff) <= 2) * 100:.3f}%, "
              f"holes {hole.sum()}, strays {stray.sum()}")
        e = np.zeros((N, N, 3), np.uint8)
        e[mh | vh] = 60
        e[hole] = [255, 40, 40]
        e[stray] = [255, 200, 0]
        Image.fromarray(np.concatenate([i.transpose(1, 0, 2)[::-1] for i in (shade(vd), shade(md), e)], 1)).save(
            os.path.join(out, f"view_{'+-'[side]}{names[a]}.png"))
print(f"error pixels in all views: {total_err}; {time.perf_counter() - t0:.1f} s")
