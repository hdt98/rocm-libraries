# hipDNN - Backend Caching Design Document

- Contributors: Adam Dickin
- **Status**: Draft

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Alternatives Considered](#6-alternatives-considered)
7. [Risks](#7-risks)
8. [Execution Plan](#8-execution-plan)
9. [Testing Plan](#9-testing-plan)
10. [Future Considerations](#10-future-considerations)
11. [Glossary](#11-glossary)

## 1. Executive Summary

This RFC proposes a caching mechanism for hipDNN that enables plugins to persist data across sessions. The design supports two complementary approaches:

1. **Callback-based caching (new plugins)**: A backend-provided callback API that plugins use to store and retrieve named data blobs. The backend owns the cache storage and provides a unified interface.
2. **Filepath-based caching (legacy plugins)**: A directory path communicated to plugins at initialization, allowing legacy libraries (e.g., MIOpen) to use their existing file-based cache mechanisms with a backend-controlled location.

The first version is scoped to file-backed caching. The callback API is designed for extensibility so that future versions could support alternative backends (e.g., in-memory, network-attached).

### Design Principles
- **Co-location**: All plugin cache data lives under a single, configurable root directory
- **Isolation**: Each plugin receives its own subdirectory to prevent conflicts
- **Legacy-friendly**: Plugins wrapping libraries with existing cache logic can direct that cache to a backend-managed path
- **Optional**: Plugins that do not need caching can ignore the mechanism entirely

## 2. Problem Statement

### 2.1 Current Limitations

hipDNN plugins currently have no mechanism to persist data between sessions. The plugin API provides:

- A logging callback (`hipdnnPluginSetLoggingCallback`)
- A HIP stream (`hipdnnEnginePluginSetStream`)
- Operation graphs and engine configurations (via flatbuffer serialization)

There is no way for the backend to communicate a cache directory, nor any callback mechanism for plugins to store or retrieve data through the backend.

### 2.2 Why This Matters

Several real-world scenarios require plugin caching:

1. **MIOpen kernel compilation caches**: MIOpen compiles GPU kernels at runtime and caches the compiled binaries. Without a backend-managed cache path, MIOpen uses its own default directory (`~/.cache/miopen/`), which is invisible to hipDNN and cannot be controlled by the user through hipDNN's API.

2. **Algorithm selection databases**: Libraries like MIOpen maintain performance databases that record which algorithm was fastest for a given problem configuration. These databases need to persist across sessions.

3. **Future plugin needs**: Compiled shader caches, auto-tuning results, serialized execution plans, and other artifacts that benefit from persistence.

### 2.3 Requirements

1. **Plugin data persistence**: Plugins must be able to store and retrieve arbitrary data that survives process restarts
2. **Backend-managed location**: The backend must control where cache data is stored, enabling co-location and user configuration
3. **Legacy library support**: Plugins wrapping libraries with existing cache mechanisms (e.g., MIOpen) must be able to redirect those caches to the backend-managed location
4. **User control**: Users must be able to configure the cache location through the frontend API
5. **Plugin isolation**: Each plugin's cache data must be isolated from other plugins

## 3. Current System Overview

### 3.1 Plugin Initialization Flow

When a hipDNN handle is created, the backend:

1. Discovers plugin shared libraries from configured paths (`hipdnnSetEnginePluginPaths_ext`)
2. Loads each plugin and queries metadata (`hipdnnPluginGetName`, `hipdnnPluginGetVersion`, etc.)
3. Sets the logging callback (`hipdnnPluginSetLoggingCallback`)
4. Creates engine plugin handles (`hipdnnEnginePluginCreate`)
5. Sets the HIP stream (`hipdnnEnginePluginSetStream`)

There is no step where cache configuration is communicated to the plugin.

### 3.2 Existing Callback Pattern

The logging system establishes the pattern for backend-to-plugin communication:

```c
// Backend sets a callback on the plugin
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback);

// Plugin calls the callback when it needs to log
callback(HIPDNN_SEV_INFO, "message");
```

The logging callback is a **process-global** function: it is called once during plugin loading in `PluginManagerBase::loadPluginFromFile()`, not per-handle. The caching configuration follows the same process-global scope — cache paths and callbacks are set during plugin loading, before any per-handle engine plugin handles are created.

### 3.3 MIOpen Provider's Current Cache Behavior

The MIOpen provider currently relies on MIOpen's internal cache management:

- MIOpen uses `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` environment variables
- Default cache location: `~/.cache/miopen/`
- Caches include: compiled kernel binaries, find-db (algorithm selection), and performance databases
- The hipDNN backend has no visibility into or control over this caching

### 3.4 Existing Kernel Cache Concepts

The backend already defines descriptor types related to kernel caching:

```c
HIPDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR = 12
HIPDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR = 13
HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE = 306
HIPDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED = 1100
```

These are for **in-memory kernel caching within execution plans** and are unrelated to the persistent file caching proposed here. The designs are complementary: kernel cache descriptors control runtime caching of compiled kernels, while this RFC addresses cross-session persistence of arbitrary plugin data.

## 4. Proposed Design

### 4.1 Overview

The caching system has three layers:

```
+-------------------+
|    Frontend API    |  User sets cache path, queries cache info
+-------------------+
         |
+-------------------+
|   Backend API      |  Manages cache directory, provides callbacks to plugins
+-------------------+
         |
+-------------------+
|   Plugin SDK API   |  Plugin receives cache path and/or store/load callbacks
+-------------------+
```

### 4.2 Cache Directory Structure

The backend manages a cache root directory with per-plugin subdirectories:

```
<cache_root>/
  hipdnn/
    <plugin_name>/           # One directory per loaded plugin
      ...                    # Plugin-managed contents (opaque to backend)
```

- `<cache_root>` defaults to a platform-appropriate location (e.g., `$XDG_CACHE_HOME/hipdnn` on Linux, `%LOCALAPPDATA%/hipdnn` on Windows) and is configurable via the frontend API or an environment variable
- The `hipdnn/` prefix under the cache root prevents collision if the cache root is shared with other applications
- Plugin subdirectories are named after the plugin name returned by `hipdnnPluginGetName()`
- The backend creates these directories as needed

### 4.3 Plugin SDK API Extensions

#### 4.3.1 Cache Path API (Legacy Support)

A new optional function in the plugin API for receiving a cache directory path:

```c
// plugin_sdk/include/hipdnn_plugin_sdk/PluginApi.h

/**
 * @brief Sets the cache directory path for this plugin.
 *
 * The backend calls this function to inform the plugin of a directory it may
 * use for file-based caching. The directory is guaranteed to exist when this
 * function is called and is exclusive to this plugin.
 *
 * This function is intended for plugins that wrap libraries with existing
 * file-based cache mechanisms (e.g., MIOpen). New plugins should prefer the
 * cache store/load callbacks instead.
 *
 * @param[in] path  Null-terminated absolute path to the plugin's cache directory.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success.
 *
 * @note This function is OPTIONAL. Plugins that do not implement it will
 *       not receive a cache path. The backend detects support via dlsym/GetProcAddress.
 * @note The plugin must not assume the directory contents persist indefinitely;
 *       the user may clear the cache at any time.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetCachePath(const char* path);
```

#### 4.3.2 Cache Callback API (New Plugins)

A callback-based API for plugins that want the backend to manage their cache storage:

```c
// plugin_sdk/include/hipdnn_plugin_sdk/PluginApiDataTypes.h

/**
 * @brief Status codes for cache operations.
 */
typedef enum
{
    HIPDNN_CACHE_STATUS_SUCCESS = 0,
    HIPDNN_CACHE_STATUS_NOT_FOUND = 1,    ///< Key does not exist in cache
    HIPDNN_CACHE_STATUS_IO_ERROR = 2,     ///< Filesystem I/O error
    HIPDNN_CACHE_STATUS_BAD_PARAM = 3,    ///< Invalid parameter (null key, etc.)
    HIPDNN_CACHE_STATUS_BUFFER_TOO_SMALL = 4, ///< Provided buffer is too small
} hipdnnCacheStatus_t;

/**
 * @brief Parameters for a cache store operation.
 *
 * Uses the structSize pattern for forward compatibility: new fields may be
 * appended in future versions. The backend checks structSize to determine
 * which fields are present.
 */
typedef struct
{
    size_t structSize;        ///< Must be set to sizeof(hipdnnCacheStoreParams_t)
    const char* key;          ///< Null-terminated key identifying the cached item
    const void* data;         ///< Pointer to the data to store
    size_t dataSize;          ///< Size of the data in bytes
} hipdnnCacheStoreParams_t;

/**
 * @brief Parameters for a cache load operation.
 *
 * Uses the two-call pattern:
 * 1. Call with data=NULL to query the size (returned in dataSize).
 * 2. Call with allocated buffer to retrieve the data.
 */
typedef struct
{
    size_t structSize;        ///< Must be set to sizeof(hipdnnCacheLoadParams_t)
    const char* key;          ///< Null-terminated key identifying the cached item
    void* data;               ///< Buffer to receive the data, or NULL to query size
    size_t* dataSize;         ///< [in/out] On input: buffer size. On output: data size
} hipdnnCacheLoadParams_t;

/**
 * @brief Parameters for a cache delete operation.
 */
typedef struct
{
    size_t structSize;        ///< Must be set to sizeof(hipdnnCacheDeleteParams_t)
    const char* key;          ///< Null-terminated key identifying the item to delete
} hipdnnCacheDeleteParams_t;

/**
 * @brief Callback to store data in the cache.
 *
 * @param[in] context  Opaque backend-owned context (from hipdnnCacheCallbacks_t).
 * @param[in] params   Store parameters.
 *
 * @return HIPDNN_CACHE_STATUS_SUCCESS on success.
 */
typedef hipdnnCacheStatus_t (*hipdnnCacheStoreCallback_t)(
    void* context,
    const hipdnnCacheStoreParams_t* params);

/**
 * @brief Callback to load data from the cache.
 *
 * @param[in]     context  Opaque backend-owned context (from hipdnnCacheCallbacks_t).
 * @param[in,out] params   Load parameters.
 *
 * @return HIPDNN_CACHE_STATUS_SUCCESS on success.
 * @return HIPDNN_CACHE_STATUS_NOT_FOUND if the key does not exist.
 * @return HIPDNN_CACHE_STATUS_BUFFER_TOO_SMALL if the buffer is too small
 *         (params->dataSize is updated to the required size).
 */
typedef hipdnnCacheStatus_t (*hipdnnCacheLoadCallback_t)(
    void* context,
    hipdnnCacheLoadParams_t* params);

/**
 * @brief Callback to delete a cached entry.
 *
 * @param[in] context  Opaque backend-owned context (from hipdnnCacheCallbacks_t).
 * @param[in] params   Delete parameters.
 *
 * @return HIPDNN_CACHE_STATUS_SUCCESS on success (also returned if key did not exist).
 */
typedef hipdnnCacheStatus_t (*hipdnnCacheDeleteCallback_t)(
    void* context,
    const hipdnnCacheDeleteParams_t* params);

/**
 * @brief Cache callbacks passed to the plugin.
 *
 * The backend creates a unique instance of this struct per plugin, with the
 * context field bound to the plugin's identity. The plugin must pass context
 * as the first argument to every callback invocation.
 *
 * The plugin must NOT inspect or modify the context pointer.
 */
typedef struct
{
    hipdnnCacheStoreCallback_t store;   ///< Store data in the cache
    hipdnnCacheLoadCallback_t load;     ///< Load data from the cache
    hipdnnCacheDeleteCallback_t remove; ///< Delete an entry from the cache
    void* context;                      ///< Opaque backend-owned context, passed to all callbacks
} hipdnnCacheCallbacks_t;
```

```c
// plugin_sdk/include/hipdnn_plugin_sdk/PluginApi.h

/**
 * @brief Sets the cache callbacks for this plugin.
 *
 * The backend calls this function during plugin loading to provide
 * store/load/delete callbacks that the plugin can use to persist data.
 * The callbacks are thread-safe and may be called from any thread at
 * any time after this function returns.
 *
 * The context field in the callbacks struct is opaque and backend-owned.
 * It encodes the plugin's identity so the backend can route cache
 * operations to the correct per-plugin storage without the plugin
 * needing to identify itself.
 *
 * @param[in] callbacks  Pointer to a structure containing the cache callbacks.
 *                       The structure is copied; the pointer does not need to
 *                       remain valid after this call.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success.
 *
 * @note This function is OPTIONAL. Plugins that do not implement it will
 *       not receive cache callbacks. The backend detects support via
 *       dlsym/GetProcAddress.
 * @note The callbacks remain valid for the lifetime of the loaded plugin.
 *       If the plugin is unloaded and reloaded, new callbacks will be provided.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetCacheCallbacks(const hipdnnCacheCallbacks_t* callbacks);
```

#### 4.3.3 Cache Key Format

Cache keys are plugin-defined strings that identify cached entries. The plugin owns the
semantic meaning of the key and is responsible for encoding all dimensions that affect
cache validity.

**Valid key characters**: Alphanumeric characters, `-`, `_`, `.`, and `/`. Forward slashes
create subdirectories in the file-backed implementation, enabling hierarchical organization.

**Maximum key length**: 255 characters.

**Rejected keys**: Keys containing `..`, leading `/` (absolute paths), null bytes, or
characters outside the allowed set.

**Plugin responsibilities**: Plugins caching GPU-compiled artifacts (e.g., hipRTC binaries)
must encode all relevant dimensions in the key, such as:
- GPU architecture (e.g., `gfx942`)
- Data types (e.g., `fp16`, `fp32`)
- Problem dimensions or configurations
- Compiler flags or optimization levels

Example keys:
```
gfx942/conv_fwd/fp16/3x3x64x128      # compiled kernel, arch-specific
tuning/heuristic_weights               # tuning data, arch-independent
gfx90a/matmul/fp32/O2                  # compiled kernel with opt level
```

The backend validates key format but does not interpret key semantics.

### 4.4 Backend API Extensions

#### 4.4.1 Cache Path Configuration

```c
// backend/include/hipdnn_backend.h

/**
 * @brief Sets the cache directory root for hipDNN.
 *
 * Configures the root directory under which plugin cache data will be stored.
 * Must be called before creating a hipDNN handle, as cache paths are
 * communicated to plugins during plugin loading (which occurs at handle
 * creation time). Follows the same constraint as hipdnnSetEnginePluginPaths_ext.
 *
 * If called while plugins are loaded (including via lazy unloading), returns
 * HIPDNN_STATUS_NOT_SUPPORTED. The caller must ensure all handles are
 * destroyed and lazy plugin references are released before changing the
 * cache path.
 *
 * If not called, the backend uses a platform-appropriate default:
 * - Linux: $XDG_CACHE_HOME/hipdnn or ~/.cache/hipdnn
 * - Windows: %LOCALAPPDATA%/hipdnn
 *
 * The directory will be created if it does not exist.
 *
 * @param[in] cachePath  Null-terminated absolute path to the cache root directory.
 *                       Pass NULL to reset to the default location.
 *
 * @retval HIPDNN_STATUS_SUCCESS           The path was set successfully.
 * @retval HIPDNN_STATUS_BAD_PARAM         cachePath is not an absolute path.
 * @retval HIPDNN_STATUS_NOT_SUPPORTED     Called while a handle is active.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR    Failed to create the directory.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetCachePath_ext(const char* cachePath);

/**
 * @brief Gets the current cache directory root for hipDNN.
 *
 * @param[out] cachePath     Buffer to receive the cache path string.
 * @param[in,out] pathSize   On input: buffer size. On output: required size
 *                           (including null terminator). Pass cachePath as NULL
 *                           to query the required size.
 *
 * @retval HIPDNN_STATUS_SUCCESS           The path was retrieved successfully.
 * @retval HIPDNN_STATUS_BAD_PARAM         pathSize is NULL.
 * @retval HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT  Buffer is too small
 *                                                     (pathSize updated).
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetCachePath_ext(char* cachePath, size_t* pathSize);

/**
 * @brief Disables caching for hipDNN.
 *
 * When caching is disabled, plugins will not receive cache paths or callbacks.
 * Must be called before creating a hipDNN handle.
 *
 * Caching can also be disabled via the environment variable:
 *   HIPDNN_DISABLE_CACHE=1
 *
 * @retval HIPDNN_STATUS_SUCCESS           Caching was disabled successfully.
 * @retval HIPDNN_STATUS_NOT_SUPPORTED     Called while a handle is active.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDisableCache_ext();
```

#### 4.4.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HIPDNN_CACHE_DIR` | Override cache root directory | Platform default |
| `HIPDNN_DISABLE_CACHE` | Set to `1` to disable caching entirely | `0` (enabled) |

Environment variables take precedence over API calls, matching the pattern used by `HIPDNN_LOG_LEVEL`. This allows operators and deployment scripts to override application-level settings without code changes. The full resolution order is: **environment variable > API call > platform default**.

### 4.5 Backend Implementation

#### 4.5.1 Plugin Initialization (Updated Flow)

Cache configuration is **process-global**, matching the logging callback pattern. It occurs during plugin loading in `PluginManagerBase::loadPluginFromFile()`, not during per-handle construction. This is important because:

- Cache paths and callbacks are set once per loaded plugin shared library
- With lazy plugin unloading (`HIPDNN_PLUGIN_UNLOAD_LAZY`), plugins persist across handle destroy/recreate cycles. The cache configuration persists with them.
- Changing the cache path (via `hipdnnSetCachePath_ext`) requires unloading and reloading plugins, following the same pattern as `hipdnnSetEnginePluginPaths_ext`.

The updated plugin loading flow (new steps marked with **NEW**):

1. Discover plugin shared libraries from configured paths
2. Load each plugin and query metadata
3. Set the logging callback
4. **NEW**: If caching is enabled:
   a. Resolve the cache root directory (env var > API setting > platform default)
   b. Create `<cache_root>/hipdnn/` if it does not exist
   c. For each plugin:
      - Create `<cache_root>/hipdnn/<plugin_name>/` if it does not exist
      - If the plugin exports `hipdnnPluginSetCachePath`: call it with the plugin's cache directory
      - If the plugin exports `hipdnnPluginSetCacheCallbacks`: create a per-plugin `CacheContext`, bind the plugin name and cache directory into it, and call `SetCacheCallbacks` with the backend's callbacks and the context
      - Both functions may be exported by the same plugin; the backend calls both if present
5. Create engine plugin handles (per-handle)
6. Set the HIP stream (per-handle)

**Note on ordering**: Step 4 (cache setup) occurs **before** step 5 (handle creation). This ordering is critical for legacy plugins like MIOpen, where the library may lazily initialize its cache paths on first use (e.g., during `miopenCreate()`). The cache path must be communicated to the plugin before any library calls that could trigger cache initialization.

#### 4.5.2 Cache Callback Implementation

The backend creates a per-plugin context that binds the plugin's identity:

```c++
// Internal to the backend (not exposed in the plugin SDK)
struct CacheContext
{
    std::string pluginName;
    std::filesystem::path cacheDir;
    std::mutex mutex;
};
```

The same callback function pointers are shared across all plugins; the per-plugin `CacheContext` (passed as `void* context`) routes operations to the correct subdirectory. The plugin never needs to identify itself — the backend already knows.

The backend implements the callbacks as file operations:

- **Store**: Write data to `<cache_root>/hipdnn/<context.pluginName>/<key>` (creating intermediate directories for `/` in keys as needed)
- **Load**: Read data from `<cache_root>/hipdnn/<context.pluginName>/<key>`
- **Delete**: Remove the file at `<cache_root>/hipdnn/<context.pluginName>/<key>`

Keys are validated against the rules in Section 4.3.3.

Thread safety requires two levels of synchronization:
1. **Intra-process**: A mutex in the `CacheContext` protects concurrent access from multiple threads within the same process
2. **Inter-process**: `fcntl`-based file locking on Linux (works on NFS), file locking APIs on Windows, for cross-process coordination

### 4.6 Frontend API Extensions

```cpp
// frontend/include/hipdnn_frontend/Cache.hpp
namespace hipdnn_frontend
{

/**
 * @brief Set the cache directory for hipDNN.
 *
 * Must be called before createHipdnnHandle(). Pass empty string to reset
 * to default.
 *
 * @param cachePath  Absolute path to the cache root directory.
 * @return Error indicating success or failure.
 */
inline Error setCachePath(const std::string& cachePath)
{
    const char* path = cachePath.empty() ? nullptr : cachePath.c_str();
    auto status = detail::hipdnnBackend()->setCachePath_ext(path);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to set cache path");
    return {};
}

/**
 * @brief Get the current cache directory for hipDNN.
 *
 * @return Pair of (cache path string, error).
 */
inline std::pair<std::string, Error> getCachePath()
{
    size_t pathSize = 0;
    auto status = detail::hipdnnBackend()->getCachePath_ext(nullptr, &pathSize);
    if (status != HIPDNN_STATUS_SUCCESS)
    {
        return {"", {ErrorCode::INTERNAL_ERROR, "Failed to query cache path size"}};
    }

    std::string path(pathSize - 1, '\0');
    status = detail::hipdnnBackend()->getCachePath_ext(path.data(), &pathSize);
    HIPDNN_RETURN_ON_BACKEND_FAILURE_WITH(status, "Failed to get cache path",
                                           std::string{});
    return {std::move(path), {}};
}

/**
 * @brief Disable caching for hipDNN.
 *
 * Must be called before createHipdnnHandle().
 */
inline Error disableCache()
{
    auto status = detail::hipdnnBackend()->disableCache_ext();
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to disable cache");
    return {};
}

} // namespace hipdnn_frontend
```

### 4.7 Plugin SDK Utilities (Future)

Higher-level utilities wrapping the cache callbacks for common patterns. These would live in the Plugin SDK as optional helpers:

```cpp
// plugin_sdk/include/hipdnn_plugin_sdk/CacheUtils.hpp (future)
namespace hipdnn_plugin_sdk
{

/// Store a file's contents into the cache under the given key.
hipdnnCacheStatus_t cacheStoreFile(const hipdnnCacheCallbacks_t* callbacks,
                                    const char* key,
                                    const char* filePath);

/// Load cached data and write it to a file.
hipdnnCacheStatus_t cacheLoadToFile(const hipdnnCacheCallbacks_t* callbacks,
                                     const char* key,
                                     const char* filePath);

/// Store a named blob with an ID (e.g., for compiled kernel binaries).
hipdnnCacheStatus_t cacheStoreBlob(const hipdnnCacheCallbacks_t* callbacks,
                                    const char* category,
                                    const char* blobId,
                                    const void* data,
                                    size_t dataSize);

/// Load a named blob by category and ID.
hipdnnCacheStatus_t cacheLoadBlob(const hipdnnCacheCallbacks_t* callbacks,
                                   const char* category,
                                   const char* blobId,
                                   void* data,
                                   size_t* dataSize);

} // namespace hipdnn_plugin_sdk
```

These utilities are deferred to a follow-up implementation. The first version ships the core callback and path APIs only.

## 5. Key Design Decisions

### 5.1 Two Cache Mechanisms (Callback + Path)

**Decision**: Provide both a callback-based API and a filepath-based API.

**Rationale**:
- Legacy libraries like MIOpen have deeply integrated file-based caching that cannot be easily refactored to use callbacks
- New plugins benefit from a callback API where the backend controls the storage implementation
- The filepath API is explicitly documented as "legacy support" to discourage new plugins from using it

**Trade-off**: Two APIs to maintain, but necessary to support the existing ecosystem.

### 5.2 Optional Plugin Functions (dlsym Detection)

**Decision**: Cache functions are optional; the backend detects support via `dlsym`/`GetProcAddress` rather than requiring a version bump.

**Rationale**:
- Maintains backward compatibility with plugins that do not need caching
- Follows precedent: plugins already have optional behavior (not all plugins implement all engine functions)
- Avoids forcing all plugins to implement stub functions

**Alternative considered**: Adding a plugin API version check (e.g., "version 1.1 requires cache functions"). Rejected because it would break existing plugins that update their version but don't need caching.

### 5.3 Cache Path Set Before Handle Creation

**Decision**: `hipdnnSetCachePath_ext` must be called before `hipdnnCreate`.

**Rationale**:
- Cache paths are communicated to plugins during handle creation
- Changing the cache path while plugins are active would create inconsistency
- Follows the same pattern as `hipdnnSetEnginePluginPaths_ext`

### 5.4 File-Backed Storage for First Version

**Decision**: The callback implementation uses simple file I/O for the first version.

**Rationale**:
- File-based storage is the simplest correct implementation
- Satisfies all current requirements (MIOpen kernel caches, performance databases)
- The callback abstraction allows swapping to a different backend (in-memory, database) in the future without changing the plugin API

### 5.5 Opaque Context for Plugin Isolation

**Decision**: The backend binds each plugin's identity into an opaque `void* context` that is passed to the plugin via `hipdnnCacheCallbacks_t`. The plugin passes this context back on every callback invocation. The plugin does not need to identify itself.

**Rationale**:
- Prevents a plugin from accessing another plugin's cache (the backend controls the namespace via the context, not the plugin)
- Follows the standard C callback pattern (cf. `qsort_r`, `pthread_create`) where state is captured in an opaque context rather than passed as explicit parameters
- Simplifies plugin code — no need to call `hipdnnPluginGetName()` and pass it on every cache operation

**Alternative considered**: Passing `pluginName` as an explicit parameter in store/load callbacks. Rejected because it relies on the plugin to identify itself honestly (honor system), and creates unnecessary coupling between the plugin and its own name at the cache API boundary.

### 5.6 Key Format and Validation

**Decision**: Cache keys are plugin-defined strings with a constrained character set. The backend validates keys but does not interpret their semantics. Plugins are responsible for encoding all dimensions that affect cache validity (GPU architecture, data types, problem sizes, etc.) into the key.

**Rationale**:
- Only the plugin knows what makes a cached entry unique (e.g., a hipRTC plugin must encode GPU arch, data type, problem size, and compiler flags — the backend cannot know these)
- Allowing `/` in keys enables natural hierarchical organization (e.g., `gfx942/conv_fwd/fp16/3x3`)
- The constrained character set prevents filesystem issues without leaking implementation details — the allowed set (`[a-zA-Z0-9._/-]`, no `..`, no leading `/`) works for files, databases, and in-memory backends alike

### 5.7 Struct-Based Callback Parameters

**Decision**: Callback parameters are passed via versioned structs with a `structSize` field, rather than as positional function arguments.

**Rationale**:
- New fields can be appended to the struct in future versions without changing the callback function signatures
- The backend checks `structSize` to determine which fields are available, enabling forward and backward compatibility
- Follows the established pattern used in Windows APIs (`cbSize` in `WNDCLASS`, etc.) and GPU driver interfaces

## 6. Alternatives Considered

### 6.1 Callback-Only API (No Filepath Support)

Require all plugins to use the callback API exclusively. Legacy libraries would need wrapper code to redirect their cache operations through the callbacks.

**Rejected because**:
- MIOpen's cache mechanism is deeply integrated into its codebase; wrapping it through callbacks would require significant changes to MIOpen itself (outside our control)
- Other legacy libraries may have similar constraints
- The filepath mechanism is simple to implement and solves the immediate problem

### 6.2 Plugin-Managed Caching (No Backend Involvement)

Let each plugin manage its own cache directory independently, controlled by plugin-specific environment variables.

**Rejected because**:
- No unified user control over cache location
- No co-location of cache data
- Users must configure each plugin separately
- No visibility for hipDNN into what plugins are caching or how much space is used

### 6.3 Database-Backed Cache

Use an embedded database (e.g., SQLite, RocksDB) as the cache backend instead of plain files.

**Rejected for first version because**:
- Adds a dependency that may not be available on all platforms
- Overkill for the current use cases (storing compiled kernel binaries and performance data)
- Can be added as an alternative backend behind the callback interface in the future

### 6.4 Plugin Name as Explicit Callback Parameter

Pass `pluginName` as a parameter in every store/load/delete callback, letting the plugin identify itself on each call.

**Rejected because**:
- Relies on the plugin to honestly identify itself (honor system)
- A buggy or malicious plugin could pass another plugin's name and access its cache
- Creates unnecessary boilerplate in plugin code (must call `hipdnnPluginGetName()` and pass the result on every cache operation)
- The opaque `void* context` pattern is the standard C approach for binding identity into callbacks

### 6.5 Positional Callback Parameters (No Struct)

Pass key, data, and size as individual function arguments instead of through a struct.

**Rejected because**:
- Adding new parameters in the future would require changing the callback function signature, which is a breaking change
- The `structSize` pattern enables forward and backward compatibility without function signature changes
- Minimal overhead — the struct is stack-allocated and the compiler can optimize access

### 6.6 Shared Memory / IPC Cache

Use shared memory segments for caching, enabling multiple processes to share cached data.

**Rejected because**:
- Significantly more complex
- File-based caching already provides cross-process sharing (multiple processes can read the same cache files)
- Shared memory introduces lifetime and cleanup challenges
- Can be considered in the future if needed

## 7. Risks

### 7.1 Cache Corruption

**Risk**: Concurrent writes to the same cache key from multiple threads or processes could corrupt data.

**Mitigation**:
- Intra-process: mutex in the per-plugin `CacheContext` serializes concurrent thread access
- Inter-process: `fcntl`-based POSIX record locks on Linux (preferred over `flock` because they work on NFS), file locking APIs on Windows
- Document that cache corruption is recoverable by clearing the cache directory
- The cache is a performance optimization, not a correctness requirement; losing cached data is acceptable

**Note on NFS**: `flock`-based advisory locks do not work on NFS, which is common in HPC/cluster environments. The implementation uses `fcntl` locks which work on NFSv4+. For older NFS versions, the cache remains best-effort.

### 7.2 Disk Space Usage

**Risk**: Plugin caches could grow unboundedly and consume significant disk space.

**Mitigation**:
- Document expected cache sizes for known plugins (e.g., MIOpen kernel cache)
- Provide a `hipdnnGetCachePath_ext` API so users can monitor/manage cache size
- Future work: add cache eviction policies or size limits

### 7.3 Permission Issues

**Risk**: The backend may not have write permissions to the default cache directory.

**Mitigation**:
- Default to user-writable locations (`$XDG_CACHE_HOME` or `$HOME/.cache`)
- Log a warning if cache directory creation fails
- Gracefully degrade: if caching cannot be initialized, continue without caching rather than failing
- Users can override with `HIPDNN_CACHE_DIR` or the API

### 7.4 Plugin Compatibility

**Risk**: Existing plugins will not implement the new optional functions.

**Mitigation**:
- Functions are optional; detection via `dlsym` means existing plugins work unchanged
- No version bump required
- Plugins can adopt caching support incrementally

## 8. Execution Plan

### Phase 1: Core Infrastructure

1. **Backend cache path management**
   - Implement `hipdnnSetCachePath_ext`, `hipdnnGetCachePath_ext`, `hipdnnDisableCache_ext`
   - Platform-appropriate default path resolution
   - Environment variable support (`HIPDNN_CACHE_DIR`, `HIPDNN_DISABLE_CACHE`)
   - Cache directory creation logic
   - Unit tests for path resolution and directory management

2. **Plugin SDK API additions**
   - Add `hipdnnPluginSetCachePath` declaration to `PluginApi.h`
   - Add cache callback types and `hipdnnPluginSetCacheCallbacks` declaration
   - Update plugin API version documentation

3. **Backend plugin initialization update**
   - Update `PluginCore` to detect optional cache symbols via `tryAssignSymbol`
   - Update plugin loading flow to call `SetCachePath` and `SetCacheCallbacks` on plugins
   - Create per-plugin `CacheContext` instances with bound plugin name and cache directory
   - Create per-plugin cache subdirectories
   - Implement file-backed store/load/delete callbacks
   - Key validation logic (Section 4.3.3 rules)
   - Thread-safe locking: per-context mutex (intra-process) + `fcntl` locks on Linux / file locking on Windows (inter-process)

### Phase 2: Frontend API

4. **Frontend wrappers**
   - Add `Cache.hpp` with `setCachePath`, `getCachePath`, `disableCache`
   - Add virtual methods to `IHipdnnBackend` and implementations to `HipdnnBackendWrapper`
   - Unit tests using fake backend

### Phase 3: MIOpen Provider Integration

5. **MIOpen provider cache path support**
   - Implement `hipdnnPluginSetCachePath` in the MIOpen provider
   - Use the provided path to set MIOpen's cache directory (via `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR` or MIOpen API)
   - Integration tests verifying cache file co-location

### Phase 4: Documentation

6. **User and developer documentation**
   - User guide for configuring cache location
   - Plugin developer guide for implementing cache support
   - Update HowTo documentation
   - Document environment variables

## 9. Testing Plan

### 9.1 Unit Tests

#### Backend Cache Management
- Default cache path resolution (Linux, Windows)
- Custom cache path via API
- Environment variable override
- Cache path validation (absolute path required)
- Directory creation (success and failure cases)
- Cache disable/enable
- Error when called with active handle

#### Cache Callbacks
- Store and load round-trip
- Load non-existent key returns `NOT_FOUND`
- Delete existing key succeeds
- Delete non-existent key succeeds (idempotent)
- Key validation rejects `..`, leading `/`, null bytes, invalid characters
- Key with `/` separators creates subdirectories and round-trips correctly
- Key at maximum length (255 chars) works
- Key exceeding maximum length is rejected
- Buffer too small returns `BUFFER_TOO_SMALL` with correct size
- Empty data store/load
- Large data store/load
- Context isolation: plugin A's callbacks cannot access plugin B's cache

#### Plugin Detection
- Plugin with cache functions detected correctly
- Plugin without cache functions works normally
- Plugin with only `SetCachePath` (no callbacks) works
- Plugin with only `SetCacheCallbacks` (no path) works

### 9.2 Integration Tests

#### MIOpen Provider
- Cache files appear in backend-managed directory
- Custom cache path redirects MIOpen's cache
- Cache survives handle destroy/recreate cycle
- Cache disable prevents cache directory creation

### 9.3 Thread Safety Tests
- Concurrent store/load from multiple threads
- Store/load during handle creation/destruction

## 10. Future Considerations

### 10.1 Cache Eviction

A future version could add cache size limits and eviction policies:
- LRU (least recently used) eviction
- Size-based limits (per-plugin or global)
- Time-based expiration

### 10.2 Cache Versioning

Plugins may need to invalidate cached data when their internal formats change. Future work could add:
- Version tags on cached entries
- Automatic invalidation on plugin version change

### 10.3 Plugin SDK Utilities

Higher-level helpers wrapping the raw callbacks (as outlined in section 4.7) for common patterns like file caching, named blobs, and categorized storage.

### 10.4 Alternative Storage Backends

The callback abstraction enables future storage backends:
- In-memory cache for testing or ephemeral workloads
- Database-backed cache for indexed lookups
- Network-attached storage for distributed environments

### 10.5 Cache Introspection API

A future API to query cache contents, sizes, and per-plugin usage, enabling monitoring and management tools.

## 11. Glossary

- **Cache root**: The top-level directory under which all hipDNN cache data is stored
- **Plugin cache directory**: A per-plugin subdirectory under the cache root (`<cache_root>/hipdnn/<plugin_name>/`)
- **Cache callback**: A function pointer provided by the backend for plugins to store/load/delete data
- **Cache context**: An opaque, backend-owned pointer that encodes the plugin's identity and cache directory; passed to all callback invocations
- **Cache key**: A plugin-defined string identifier for a cached data entry. May contain `/` for hierarchical organization. The plugin is responsible for encoding all cache-relevant dimensions (GPU arch, data type, etc.) into the key
- **Legacy plugin**: A plugin wrapping a library with existing file-based cache mechanisms that cannot easily use the callback API
- **Co-location**: Storing all plugin cache data under a single directory tree for unified management
- **Key validation**: Verification that cache keys conform to the allowed character set and do not contain path traversal sequences
