# Kpack Multi-Architecture Hello World

## Motivation

[Kpack](https://github.com/ROCm/TheRock/blob/develop/docs/rfcs/rfc0008_kpack.md) (kernel pack) is a binary archive format that separates GPU device code from host code for efficient multi-architecture distribution. Instead of shipping fat binaries that embed kernels for every supported GPU, kpack packages per-architecture `.hsaco` code objects into compressed archives that are loaded at runtime based on the detected GPU.

We want to understand how kpack can work with CK Tile kernels. The challenge is that kpack assumes a clean separation between host and device code, but CK Tile's current design doesn't have that separation:

- **Architecture-dependent host code**: CK Tile kernel objects contain host-side logic (tile sizes, pipeline configuration, launch parameters) that varies by GPU architecture. The host code is not architecture-independent.
- **Coupled launch and device code**: Kernel launches happen inside the same template-instantiated objects that contain the device code. There is no standalone device-only kernel file that can be compiled in isolation.

This experimental demo starts with a simple vector-add kernel where the host/device separation is trivial, so we can validate the kpack build and load pipeline end-to-end. The real work is figuring out how to satisfy kpack's requirements given CK Tile's current object design.

## Open Questions for CK Tile Integration

- How do we separate device code compilation from the host-side kernel objects that currently contain both?
- Can the architecture-dependent host logic (tile sizes, pipeline selection) be made architecture-independent, or do we need per-arch host code too?
- How does `hipModuleLoadData` / `hipModuleGetFunction` interact with CK Tile's current kernel launch patterns?
- What changes to the CK Tile kernel instantiation layer are needed to support runtime code object loading?

## Pipeline

```
vector_add.hip
    | (clang++ --cuda-device-only per arch)
    v
vector_add_gfx90a.hsaco
vector_add_gfx942.hsaco
vector_add_gfx950.hsaco
    | (pack.py)
    v
kernels.kpack  (single archive, multiple architectures)
    | (loaded at runtime by kpack C API)
    v
kpack_hello_world  (demo executable, uses hipModuleLoadData)
```

## Directory Structure

```
experimental/kpack/
├── CMakeLists.txt                  # Top-level (delegates to example)
├── README.md
├── rocm_kpack/                     # Vendored kpack C runtime library (from TheRock)
│   ├── CMakeLists.txt
│   ├── include/rocm_kpack/
│   │   ├── kpack.h
│   │   ├── kpack_types.h
│   │   └── kpack_export.h
│   └── src/
│       ├── kpack_internal.h
│       ├── isa_target_match.h
│       ├── kpack.cpp
│       ├── archive.cpp
│       ├── compression.cpp
│       ├── toc_parser.cpp
│       ├── loader.cpp
│       ├── path_resolution.cpp
│       └── isa_target_match.cpp
└── examples/
    └── hello_world/
        ├── CMakeLists.txt          # Standalone build
        ├── vector_add.hip          # Device kernel (compiled per-arch)
        ├── pack.py                 # Packs .hsaco files into .kpack archive
        └── main.cpp                # Demo: open archive, detect GPU, load & run kernel
```

## Dependencies

- **ROCm** with HIP support (`/opt/rocm`)
- **System packages**: `libmsgpack-cxx-dev`, `libzstd-dev`
- **Python packages**: `msgpack` (`pip install msgpack`)

## Build

The hello_world example is standalone — build it directly:

```bash
cd experimental/kpack/examples/hello_world

cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"

ninja -C build
```

## Run

On a machine with a supported GPU:

```bash
./build/kpack_hello_world build/kernels.kpack
```

Expected output:

```
Opened build/kernels.kpack — architectures: gfx90a, gfx942
Detected GPU: gfx942
Loaded kernel: 41200 bytes
Running vectorAdd (N=1024)...
Verification PASSED!
```

If the current GPU's architecture is not in the archive, the demo prints a clear error and exits.

## Inspect the Archive

```bash
python3 -c "
import msgpack, struct
f = open('build/kernels.kpack', 'rb')
magic, version, toc_offset = struct.unpack('<4sIQ', f.read(16))
print(f'Magic: {magic}, Version: {version}, TOC offset: {toc_offset}')
f.seek(toc_offset)
toc = msgpack.unpack(f, raw=False)
for binary, archs in toc['toc'].items():
    for arch, meta in archs.items():
        print(f'  {binary}/{arch}: {meta[\"original_size\"]} bytes')
"
```

## How It Works

1. **Compile**: `clang++ --cuda-device-only` compiles `vector_add.hip` into per-architecture `.hsaco` code objects
2. **Pack**: `pack.py` concatenates the `.hsaco` blobs after a 16-byte KPAK header, then appends a MessagePack table of contents (TOC) recording each blob's offset, size, and architecture
3. **Load**: `main.cpp` opens the archive via the kpack C API, queries the detected GPU's architecture, extracts the matching code object, and loads it into HIP via `hipModuleLoadData`

## Kpack Archive Format

```
[0x00]  "KPAK"              4 bytes   Magic
[0x04]  version             4 bytes   Little-endian uint32 (currently 1)
[0x08]  toc_offset          8 bytes   Little-endian uint64
[0x10]  blob_0              variable  Raw .hsaco for first arch
        blob_1              variable  Raw .hsaco for second arch
        ...
[toc_offset]  MessagePack TOC     variable  Compression scheme, arch list, blob metadata, nested TOC
```

## Vendored Runtime

The `rocm_kpack/` directory contains a local copy of the kpack C runtime library, taken from [TheRock](https://github.com/ROCm/TheRock) (`base/rocm-kpack/runtime/`) for this experimental demo. This avoids requiring a full TheRock build as a dependency -- the runtime is small and self-contained. Once kpack ships as part of a ROCm release, this vendored copy should be replaced with a proper `find_package(rocm_kpack)`.

It provides:

- `kpack_open` / `kpack_close` -- archive lifecycle
- `kpack_get_architecture_count` / `kpack_get_architecture` -- enumerate architectures
- `kpack_get_binary_count` / `kpack_get_binary` -- enumerate binaries
- `kpack_get_kernel` / `kpack_free_kernel` -- extract a code object by binary name + architecture
- NoOp and Zstd per-kernel decompression
- Higher-level loader/cache API for runtime integration
