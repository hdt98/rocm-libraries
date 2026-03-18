# Insights: The kpack Loading Model

This example establishes the end-to-end kpack pipeline. Every kernel operation in rocm_ck will follow this pattern.

## 1. The Five-Stage Loading Pipeline

The consumer-facing API surface is a five-stage pipeline:

1. **Open archive** — `kpack_open(path, &archive)` reads the header and TOC
2. **Detect GPU** — `hipGetDeviceProperties` → strip feature flags → base arch (e.g., `"gfx942"`)
3. **Lookup kernel** — `kpack_get_kernel(archive, name, arch, &data, &size)` finds the matching blob
4. **Load module** — `hipModuleLoadData(&module, data)` → `hipModuleGetFunction(&fn, module, symbol)`
5. **Launch** — `hipModuleLaunchKernel(fn, grid, block, ..., launch_config)`

This is the entire integration surface. A rocm_ck consumer needs exactly these five steps to go from an archive on disk to a running kernel. Everything else — compilation, packing, variant selection — happens at build time.

## 2. hipModuleLoadData Is the Integration Point

kpack produces raw `.hsaco` bytes. HIP's module API consumes them. This is a clean, stable boundary maintained by AMD. No custom loaders, no ELF parsing, no architecture-specific dispatch logic on the consumer side.

The kpack library's only job is to get the right bytes to `hipModuleLoadData`. Everything upstream (archive format, compression, TOC indexing) is an implementation detail hidden behind `kpack_get_kernel`.

## 3. Argument Passing via HIP_LAUNCH_PARAM

`hipModuleLaunchKernel` requires struct-based argument packing with explicit layout:

```cpp
struct {
    const float* a;
    const float* b;
    float* c;
    int n;
} kernel_args = {device_a, device_b, device_result, n};

size_t kernel_args_size = sizeof(kernel_args);
void* launch_config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kernel_args_size,
    HIP_LAUNCH_PARAM_END
};
```

The struct layout is a binary contract between host and device — field order, sizes, alignment, and padding must match exactly. Example 03 takes this further with `static_assert` validation on size, alignment, and offsets. Every rocm_ck kernel must follow the same discipline.

## 4. Per-Architecture Compilation + Runtime Selection

Single binary distribution for multiple GPU architectures. Each architecture gets a separately compiled `.hsaco`. The kpack TOC maps `(kernel_name, arch)` to the correct blob. At runtime, the host detects the GPU and retrieves the matching code object.

This eliminates the fat binary problem — no `--offload-arch` list baked into the host executable. The archive carries all supported architectures; the host binary is architecture-agnostic.

## 5. The kpack Archive Format Is Simple by Design

The format is three sections:

| Section | Content |
|---------|---------|
| Header (16 bytes) | Magic `"KPAK"`, version (uint32), TOC offset (uint64) |
| Blobs | Concatenated `.hsaco` binaries |
| TOC | MessagePack: `toc[kernel_name][arch]` → `{ordinal, original_size, type}` |

No compression in the POC. The `compression_scheme` field in the TOC enables future extension (Zstd support is already implemented) without breaking existing readers — unknown schemes produce a clear error, not silent corruption.

> **Production implication**: The format is deliberately extensible. New fields in the TOC (variant metadata, tuning parameters) are ignored by readers that don't understand them. This forward-compatibility is essential for a format that will evolve with the kernel library.

## 6. Memory Ownership Is Explicit

`kpack_get_kernel` allocates a copy of the kernel bytes via `malloc`; the caller frees via `kpack_free_kernel` (which calls `free`). The returned pointer survives `kpack_close`.

This is the right ownership model for production:
- No dangling references to archive-internal buffers
- Caller controls lifetime — can close the archive and keep the kernel bytes
- Thread-safe: the archive's internal kernel cache is protected by a mutex, but each caller gets an independent copy

The alternative — returning a pointer into the archive's internal buffer — would create lifetime coupling and thread-safety issues. The copy cost is negligible compared to `hipModuleLoadData`.