# RFC: Engine Selection and Heuristics Framework for hipDNN

| Field | Value |
|-------|--------|
| Status | First draft |
| Component | Backend, heuristic/selection plugins |

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Motivation](#2-motivation)
3. [Baseline: current behavior](#3-baseline-current-behavior)
4. [Goals and non-goals](#4-goals-and-non-goals)
5. [Design overview](#5-design-overview)
6. [Device properties](#6-device-properties)
7. [SelectionEngine interface](#7-selectionengine-interface)
8. [C ABI for heuristic plugins](#8-c-abi-for-heuristic-plugins)
9. [Policy plugins and the outer loop](#9-policy-plugins-and-the-outer-loop)
10. [HeuristicPluginManager and resource layer](#10-heuristicpluginmanager-and-resource-layer)
11. [Versioning and compatibility checks](#11-versioning-and-compatibility-checks)
12. [Logging](#12-logging)
13. [Serialized graph and graph-level preferences](#13-serialized-graph-and-graph-level-preferences)
14. [EngineHeuristicDescriptor integration](#14-engineheuristicdescriptor-integration)
15. [Public API notes](#15-public-api-notes)
16. [Testing](#16-testing)
17. [Risks and open questions](#17-risks-and-open-questions)
18. [Glossary](#18-glossary)

---

## 1. Executive summary

This document proposes a **first-draft** design for an extensible **engine selection and heuristics** framework in hipDNN. The framework replaces the hard-coded ordering in `EngineHeuristicDescriptor::finalize()` with an **ordered outer loop** over **policy plugins**. Each policy is a distinct selection strategy (for example: machine-learning-based selector, rule-based selector, cache-driven selector, or a deterministic default selector).

Device capabilities enter the system **only as explicit data** (`DeviceProperties`), not by probing HIP from inside plugins. Callers may later supply overrides via backend attributes (for example `HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`), aligned with the broader direction of **device properties instead of implicit handle coupling**.

The **`SelectionEngine`** C++ type is a **facade** used inside the backend; dynamically loaded heuristic plugins implement a **stable C ABI** ([§8](#8-c-abi-for-heuristic-plugins)) with C linkage, POD structs, and opaque instance handles. That ABI is **separate from the engine plugin ABI**—a `.so` is one or the other, never both.

**HeuristicPluginManager** (and a handle-scoped **HeuristicPluginResourceManager** modeled on **`EnginePluginResourceManager`**) own discovery, loading, version validation, and registration of heuristic shared libraries. **`EngineHeuristicDescriptor::finalize()`** obtains candidate engines from the existing **engine** resource manager and obtains **`SelectionEngine`** instances from the **heuristic** resource manager for each entry in the **ordered policy list**, which is **user-configurable** with default **`{ "SelectionEngine::Config", "SelectionEngine::Fallback" }`** ([§5.3](#53-ordered-policy-list-default-and-user-configuration)).

---

## 2. Motivation

- **Extensibility:** Today, engine ordering is fixed in backend code (`utilities::sortEngineIds`). New strategies (performance models, user policy, caches) require editing the core library.
- **Explicit device context:** Heuristics should depend on **passed-in** device facts, matching hipDNN’s reserved device-property attributes and reducing reliance on global HIP state or opaque handle internals for device semantics.
- **Consistency:** Engine plugins already load from trusted paths and validate API versions. Heuristic policies should follow the **same class of patterns** (shared libraries, version gates) **without sharing symbols or types** with engine plugins.
- **Interop:** A **C ABI** for heuristic plugins enables non–C++ consumers, stable binary boundaries, and future P/Invoke-style bindings without dragging in the engine plugin surface.

---

## 3. Baseline: current behavior

`EngineHeuristicDescriptor` today:

1. Obtains candidate engine IDs from `EnginePluginResourceManager::getApplicableEngineIds(opGraph)`.
2. Sorts them with `utilities::sortEngineIds` inside `finalize()` (for example prioritizing MIOPEN engine IDs and deprioritizing deterministic variants).
3. Exposes the ordered list through `HIPDNN_ATTR_ENGINEHEUR_RESULTS` as engine configuration descriptors.

This RFC preserves **equivalent deterministic ordering** as one **policy** (the “default” or “fallback” selector), implemented as a plugin or thin wrapper, not as unconditional backend logic.

---

## 4. Goals and non-goals

**Goals**

- **Outer-loop-only** orchestration: try an ordered list of **policy** plugins until one succeeds.
- **Explicit `DeviceProperties`** at the selection boundary; plugins do not call `hipGetDevice` / `hipGetDeviceProperties` themselves.
- **Pluggable policies** loaded through a **`HeuristicPluginManager`** / **`HeuristicPluginResourceManager`** layer that mirrors **engine** plugin *mechanics* (paths, discovery, version checks, optional static path configuration) but uses the **heuristic-only** C ABI in [§8](#8-c-abi-for-heuristic-plugins).
- **Stable C ABI** for heuristic `.so` files: module metadata, instance lifecycle, and selection entry points as in [§8](#8-c-abi-for-heuristic-plugins) (to be codified in headers such as **`HeuristicsPluginApi.h`**).
- **Same logging path** as the rest of the backend (see [§12 Logging](#12-logging)).
- **Version compatibility** at load time (see [§11](#11-versioning-and-compatibility-checks)).
- **User-configurable policy order** with a documented **default** list ([§5.3](#53-ordered-policy-list-default-and-user-configuration)).

**Non-goals (this draft)**

- Inner “stage chains” inside a single policy (for example PerEngine → Config → Fallback as mandatory sub-steps). Policy composition is expressed by **ordering multiple policy plugins** in the outer list, not by nested pipelines inside `SelectionEngine`.
- Mandating async selection in v1; the API is only **shaped** to allow it later (including the two-phase C selection API in [§8.8](#88-selection-maps-to-sortengineids--sortedengineids)).

---

## 5. Design overview

### 5.1 Single orchestration model: outer loop

There is **one** primary control flow:

1. Build the list of **candidate engine IDs** from existing **engine** plugins (unchanged).
2. Obtain **device properties** (see [§6](#6-device-properties)) and the **serialized operation graph** (see [§13](#13-serialized-graph-and-graph-level-preferences)).
3. Resolve the **ordered list of policy plugin IDs** ([§5.3](#53-ordered-policy-list-default-and-user-configuration)).
4. For each policy ID, obtain a **`SelectionEngine`** from **`HeuristicPluginResourceManager`** (which delegates to **`HeuristicPluginManager`**-loaded plugins), configure it with `DeviceProperties`, then:
   - If `isApplicable` is false, continue.
   - If `sortEngineIds` returns true, take `sortedEngineIds` and **stop** (no later policies run).
5. If no policy succeeds, apply a **built-in final fallback** that mirrors today’s `utilities::sortEngineIds` (or invoke the default policy plugin if it is guaranteed to always succeed).

There is **no** separate inner registry of sub-stages inside `SelectionEngine` in this design. If a team wants “config then fallback” inside one deliverable, they ship **one** policy plugin that implements that sequence internally, or they register two entries in the **outer** list.

### 5.2 Policy examples (illustrative)

| Policy (illustrative name) | Role |
|----------------------------|------|
| **Cache selector** | Reuse a persisted ordering keyed by graph fingerprint and device properties. |
| **ML selector** | Score candidates using a model or cost function. |
| **Rule selector** | Deterministic rules over graph features and `DeviceProperties`. |
| **Default selector** | Reproduce current `sortEngineIds` behavior; typically last registered or used as ultimate fallback. |

These are **examples**; the registry uses stable string IDs chosen by each plugin (and exposed via **`hipdnnHeuristicGetPolicyId`** in [§8](#8-c-abi-for-heuristic-plugins)).

### 5.3 Ordered policy list: default and user configuration

The outer loop needs an ordered list **`orderedPolicyIds`**: each element is a **string** that must match **`hipdnnHeuristicGetPolicyId`** for a loaded heuristic plugin (or a **built-in** policy registered under the same ID without a separate `.so`).

#### 5.3.1 Well-known policy IDs

This draft standardizes two logical policies using **well-known string IDs** (the `SelectionEngine::` prefix is a **naming convention** in the string itself, not C++ language linkage or a shared type with the C++ `SelectionEngine` class):

| Policy ID string | Role |
|------------------|------|
| **`SelectionEngine::Config`** | Applies user / graph configuration (for example honoring **`preferred_engine_id`**, env-based disables, future descriptor knobs). Typically runs **first** so explicit intent overrides later policies. |
| **`SelectionEngine::Fallback`** | Deterministic ordering aligned with today’s **`utilities::sortEngineIds`** (for example MIOPEN preference, deterministic engine last). Typically runs **after** Config so there is always a compatible ordering when Config does not fully decide. |

Shipped heuristic plugins (or built-in adapters) **must** export these exact strings from **`hipdnnHeuristicGetPolicyId`** when they implement the Config and Fallback behaviors.

#### 5.3.2 Default `orderedPolicyIds`

If the user does **not** override the list, the backend uses:

```text
{ "SelectionEngine::Config", "SelectionEngine::Fallback" }
```

So **Config** is tried first; if it does not win the outer loop (not applicable or `sortEngineIds` does not report success), **Fallback** runs and is expected to **always** succeed for valid candidate sets.

#### 5.3.3 How the user sets `orderedPolicyIds`

**Resolution order (highest precedence first):**

1. **Engine-heuristic descriptor (per finalize)** — User sets an explicit ordered list via a backend attribute (proposal: **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER`** or extension equivalent: array of UTF-8 policy ID strings). Applies only to that **`EngineHeuristicDescriptor`** instance.
2. **hipDNN handle (per handle)** — Optional API on **`HeuristicPluginResourceManager`** or a handle extension (proposal: **`hipdnnSetHeuristicPolicyOrder_ext(handle, ids, count)`** or C++ **`setDefaultHeuristicPolicyOrder`**). Used when the descriptor has no override.
3. **Process environment (optional)** — Proposal: **`HIPDNN_HEURISTIC_POLICY_ORDER`** as a comma-separated list of policy IDs, applied when neither descriptor nor handle supplied a list. Exact syntax and escaping TBD.
4. **Built-in default** — **`{ "SelectionEngine::Config", "SelectionEngine::Fallback" }`** when nothing above applies.

The implementation should **validate** each ID against loaded/built-in policies (and log or skip unknown IDs per policy TBD).

#### 5.3.4 Relationship to `finalize()`

`EngineHeuristicDescriptor::finalize()` (or a helper on **`HeuristicPluginResourceManager`**) **merges** the resolution order above into a single **`std::vector<std::string> orderedPolicyIds`** before the outer loop in [§14.2](#142-pseudocode-for-finalize-first-draft).

---

## 6. Device properties

### 6.1 `DeviceProperties` struct

A single, plain structure carries the device facts needed for selection. It is filled by the backend (or by a future attribute override) and passed into every policy invocation.

```cpp
struct DeviceProperties
{
    int    deviceId            = -1;
    int    multiProcessorCount = 0;
    size_t totalGlobalMem      = 0;
    // Optional later: architecture name / ID, wavefront size, etc.
};
```

The C ABI uses the POD layout **`hipdnnHeuristicDeviceProperties_t`** in [§8.4](#84-deviceproperties-pod); the backend maps between C++ and C representations.

### 6.2 Default acquisition: `queryDeviceProperties()`

The backend may implement a helper (in an anonymous namespace or a small utility unit) that maps from HIP:

```cpp
namespace
{
DeviceProperties queryDeviceProperties()
{
    DeviceProperties out{};
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
    {
        return out;
    }
    out.deviceId = device;

    hipDeviceProp_t hipProps{};
    if(hipGetDeviceProperties(&hipProps, device) != hipSuccess)
    {
        return out;
    }
    out.multiProcessorCount = hipProps.multiProcessorCount;
    out.totalGlobalMem      = static_cast<size_t>(hipProps.totalGlobalMem);
    return out;
}
} // namespace
```

**Important:** Plugins **must not** call this helper; they consume only the `DeviceProperties` instance passed into them via `SelectionEngine::setConfig` (or the C **`hipdnnHeuristicEngineSetConfig`** in [§8](#8-c-abi-for-heuristic-plugins)).

### 6.3 Proposed override: descriptor-level device properties

**First-draft proposal:** Extend the engine-heuristic descriptor so callers can optionally supply device properties explicitly, using the already-reserved attribute space (for example `HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP` and/or a dedicated `HIPDNN_BACKEND_DEVICEPROP_DESCRIPTOR` type when implemented).

**Semantics (proposal):**

- If the heuristic descriptor has **no** device-property override set, `EngineHeuristicDescriptor::finalize()` uses `queryDeviceProperties()` once.
- If the user sets device properties on the descriptor, **that** structure is used instead of querying HIP in finalize.

This keeps **one** pathway into heuristics (`DeviceProperties`) while allowing tests, multi-GPU hosts, and API parity with “device props, not handle” designs.

---

## 7. SelectionEngine interface

`SelectionEngine` is a **C++ facade** used by the backend when driving a single policy. It is **intentionally stateful**: `sortEngineIds` may compute asynchronously in a future revision; the result is read via `sortedEngineIds`. A typical implementation holds an opaque **`hipdnnHeuristicEngine_t`** (see [§8.3](#83-opaque-instance-handle)) and forwards to the C ABI.

```cpp
class SelectionEngine
{
public:
    // Binds device properties for subsequent calls. Logging is not passed here—same as engine
    // plugins: the heuristic .so registers hipdnnHeuristicSetLoggingCallback at load time ([§12](#12-logging)).
    void setConfig(const DeviceProperties& cfg);

    // True if this policy should run for the given graph and configuration.
    bool isApplicable(const SerializedGraph& serializedGraph);

    // Runs selection; on success, stores the ordered engine IDs internally.
    bool sortEngineIds(std::vector<int64_t>& engineIds,
                       const SerializedGraph& serializedGraph);

    // Returns the result of the last successful sortEngineIds.
    std::vector<int64_t> sortedEngineIds();
};
```

**Notes**

- **Logging** matches **engine** plugins: not part of these method signatures; see [§12](#12-logging).
- `SerializedGraph` is a logical view of **pointer + size** into the FlatBuffer bytes owned by the operation graph (see [§13](#13-serialized-graph-and-graph-level-preferences)).
- Keeping **`sortEngineIds` + `sortedEngineIds` separate** (instead of returning the vector directly) preserves room for **async** work later without changing the plugin contract; the C ABI mirrors this with **`SortEngineIds`** + **`GetSortedEngineIds`** ([§8.8](#88-selection-maps-to-sortengineids--sortedengineids)).

---

## 8. C ABI for heuristic plugins

This section defines the **stable C-language ABI** for heuristic (selection policy) shared libraries. It is **orthogonal** to the **engine** plugin ABI: a single `.so` is either an **engine** plugin or a **heuristic** plugin, **not both**. Heuristic libraries **do not** export `hipdnnPluginGetName`, `hipdnnPluginGetType`, or any **engine** entry points from `PluginApi.h` / `EnginePluginApi.h`—only the symbols below (final names live in **`HeuristicsPluginApi.h`** or equivalent).

### 8.1 Design principles

- All exported symbols use **C linkage** (`extern "C"` from C++ implementations).
- Versioning and rejection of incompatible plugins follow the same **ideas** as engine plugins (major API compatibility), but the **API version string** and **symbol set** are **heuristic-specific** ([§11](#11-versioning-and-compatibility-checks)).
- Plugins **must not** call HIP, mutate hipDNN graph descriptors, or execute engines; they only **read** inputs described here and **write** reordered engine IDs.
- The host (backend) owns output buffers unless the API explicitly transfers ownership.
- **Logging** is configured only via **`hipdnnHeuristicSetLoggingCallback`** / **`SetLogLevel`** at **module load time** ([§8.2](#82-plugin-module-metadata)); instance functions do not take a logger argument ([§12](#12-logging)).

### 8.2 Plugin module metadata

Each heuristic `.so` exports the following (names are illustrative; implementations use the `hipdnnHeuristic` prefix in headers).

| Function | Purpose |
|----------|---------|
| `hipdnnHeuristicGetApiVersion(const char** version)` | Semantic version of **this C ABI** (for example `"1.0.0"`). Host rejects load on **major** mismatch. |
| `hipdnnHeuristicGetPolicyId(const char** policy_id)` | Stable string ID matched against the ordered policy list (for example **`SelectionEngine::Fallback`** or a vendor-specific ID; see [§5.3](#53-ordered-policy-list-default-and-user-configuration)). |
| `hipdnnHeuristicGetPluginVersion(const char** version)` | Plugin implementation version (informational). |
| `hipdnnHeuristicSetLoggingCallback(hipdnnCallback_t cb)` | Registers the consumer logging callback; optional `hipdnnHeuristicSetLogLevel(hipdnnSeverity_t)` mirroring engine plugin behavior. |
| `hipdnnHeuristicGetLastErrorString(const char** msg)` | Per-thread last error after a failed call; pointer valid only for immediate use (same contract as `hipdnnPluginGetLastErrorString`). |

The host identifies heuristic libraries by **which loader** opened them and by successful resolution of **`hipdnnHeuristicGetApiVersion`** (and related required symbols), not by reusing **`hipdnnPluginGetType`**.

### 8.3 Opaque instance handle

Each logical **`SelectionEngine`** instance on the host maps to one opaque plugin instance:

```c
typedef struct hipdnnHeuristicEngine_opaque* hipdnnHeuristicEngine_t;
```

### 8.4 `DeviceProperties` (POD)

Fixed-layout struct for ABI stability (the C++ `DeviceProperties` in [§6](#6-device-properties) maps to/from this):

```c
typedef struct hipdnnHeuristicDeviceProperties_t {
    int32_t device_id;                 /* -1 if unknown */
    int32_t multi_processor_count;
    uint64_t total_global_mem_bytes;
    /* Reserved: set to zero; future minor ABI may add fields at end with version bump */
    uint64_t reserved[4];
} hipdnnHeuristicDeviceProperties_t;
```

### 8.5 Status codes

Define **`hipdnnHeuristicStatus_t`** with explicit values, for example: `SUCCESS`, `ERROR`, `NOT_APPLICABLE`, `UNSUPPORTED`, `INVALID_PARAM`, `BUFFER_TOO_SMALL`. This avoids overloading **`hipdnnPluginStatus_t`** (engine plugin semantics). Each C function returns **`hipdnnHeuristicStatus_t`** unless noted.

### 8.6 Instance lifecycle

```c
hipdnnHeuristicStatus_t hipdnnHeuristicEngineCreate(hipdnnHeuristicEngine_t* out_engine);
hipdnnHeuristicStatus_t hipdnnHeuristicEngineDestroy(hipdnnHeuristicEngine_t engine);
```

- **`Create`** allocates plugin-side state; **`Destroy`** frees it. One instance per outer-loop attempt is sufficient; the host may reuse one instance across calls only if documented **thread-safe** (default: **not** thread-safe).

### 8.7 Configuration and applicability

```c
hipdnnHeuristicStatus_t hipdnnHeuristicEngineSetConfig(
    hipdnnHeuristicEngine_t engine,
    const hipdnnHeuristicDeviceProperties_t* props);

hipdnnHeuristicStatus_t hipdnnHeuristicEngineIsApplicable(
    hipdnnHeuristicEngine_t engine,
    const uint8_t* serialized_graph,
    size_t serialized_graph_size,
    int32_t* out_applicable);   /* 1 = applicable, 0 = not */
```

### 8.8 Selection (maps to `sortEngineIds` / `sortedEngineIds`)

**Two-phase** selection (matches stateful `SelectionEngine`; allows future async without renaming symbols):

```c
/* On success, may set *out_success to 1 and store result inside engine.
   Logical "no ordering produced" may use *out_success == 0 without error. */
hipdnnHeuristicStatus_t hipdnnHeuristicEngineSortEngineIds(
    hipdnnHeuristicEngine_t engine,
    int64_t* inout_engine_ids,
    size_t in_engine_id_count,
    const uint8_t* serialized_graph,
    size_t serialized_graph_size,
    int32_t* out_success);

/* After SortEngineIds with *out_success == 1; host supplies buffer. */
hipdnnHeuristicStatus_t hipdnnHeuristicEngineGetSortedEngineIds(
    hipdnnHeuristicEngine_t engine,
    int64_t* out_ids,
    size_t out_capacity,
    size_t* out_count);
```

**Contract**

- **`inout_engine_ids` / `in_engine_id_count`:** candidate engine IDs from **`EnginePluginResourceManager`**; the plugin **must** output a **permutation or subset** of this set (the host validates before accepting).
- **`out_count` == `out_capacity`:** on success, number of IDs written.
- If **`SortEngineIds`** returns **`SUCCESS`** with **`out_success == 0`**, the host treats the policy as non-winning and continues the outer loop.

### 8.9 Host integration (C++ backend)

**`HeuristicPlugin`** resolves the symbols above via `dlsym` (or the platform equivalent). **`HeuristicPluginResourceManager::createSelectionEngine(policyId)`** returns a C++ **`SelectionEngine`** that:

1. Calls **`hipdnnHeuristicEngineCreate`**.
2. Forwards **`setConfig` / `isApplicable` / `sortEngineIds` / `sortedEngineIds`** to the C API (no logger parameters—see [§12](#12-logging)).
3. Calls **`hipdnnHeuristicEngineDestroy`** in its destructor.

On library load—**before** instance **`Create`** is used—the host calls **`hipdnnHeuristicSetLoggingCallback`** (and optionally **`SetLogLevel`**) on the loaded module, same timing as engine plugins ([§12](#12-logging)).

### 8.10 ABI evolution

- **Patch/minor:** additive optional functions, or use of **`reserved`** fields with version negotiation.
- **Major:** breaking struct layout or required function set; incompatible plugins fail **`validateBeforeAdding`**-style checks at load time ([§11](#11-versioning-and-compatibility-checks)).

---

## 9. Policy plugins and the outer loop

Each **heuristic policy plugin** implements one selection strategy and is loaded by **`HeuristicPluginManager`** (see [§10](#10-heuristicpluginmanager-and-resource-layer)). The backend maintains an **ordered list** of **policy plugin IDs** (strings). For each entry, the backend:

1. Asks **`HeuristicPluginResourceManager`** to **create** a `SelectionEngine` for that policy ID (or skips the ID if unknown / failed to load—policy TBD).
2. Calls `setConfig(devProps)`.
3. Calls `isApplicable(serializedGraph)`; if false, **continue** with the **original** candidate engine list unchanged.
4. Copies candidate IDs to a working vector, calls `sortEngineIds`; on success, replaces candidates with `sortedEngineIds` and **breaks**.
5. On failure or inapplicability, continues; the next policy always starts from the **original** candidate list (unless a future RFC defines chaining semantics).

**First-success wins:** The first applicable policy that reports success defines the final ordering. Later policies are not consulted.

**Read-only contract:** Heuristic plugins **must not** mutate hipDNN graph state, engine plugin handles, device memory, or global HIP device selection. They only **read** `SerializedGraph` bytes, `DeviceProperties`, and the **candidate engine ID list** they receive, and they **output** a reordered subset or permutation of those candidates (subject to backend validation—see [§17](#17-risks-and-open-questions)).

---

## 10. HeuristicPluginManager and resource layer

This section mirrors the **two-layer** pattern used for engines today: a **manager** that loads and validates shared libraries, and a **resource manager** held by the handle that exposes high-level operations to the rest of the backend.

**Engine stack (today, for reference):**

- **`EnginePluginManager`** — searches paths, loads `.so` files, runs **`validateBeforeAdding`** (API major, duplicate engine IDs), owns **`EnginePlugin`** instances.
- **`EnginePluginResourceManager`** — constructed with `std::shared_ptr<EnginePluginManager>`; provides **`getApplicableEngineIds`**, execution, workspace queries, stream propagation, etc.; obtained from the handle via **`getPluginResourceManager()`**.

**Heuristic stack (proposal):**

### 10.1 `HeuristicPluginManager`

Analogous to **`EnginePluginManager`**:

- Extends the same **plugin manager base** pattern (shared library load, symbol resolution, lifecycle).
- Uses a **separate search path** from engine plugins (for example `hipdnn_plugins/heuristics/` and/or a dedicated env var such as `HIPDNN_HEURISTIC_PLUGIN_DIR`—exact names TBD).
- Resolves **heuristic-only** symbols from [§8](#8-c-abi-for-heuristic-plugins); does **not** use **`EnginePlugin`** symbol tables.
- Implements **`validateBeforeAdding`**-style checks before accepting a plugin: **API major** match (via **`hipdnnHeuristicGetApiVersion`**), unique **policy ID** (via **`hipdnnHeuristicGetPolicyId`**), and any additional rules from [§11](#11-versioning-and-compatibility-checks).
- Owns **`HeuristicPlugin`** wrappers that bind the C ABI and expose **`createSelectionEngine()`**-time factory behavior to the resource manager.

### 10.2 `HeuristicPluginResourceManager`

Analogous to **`EnginePluginResourceManager`**:

- Holds **`std::shared_ptr<HeuristicPluginManager> _pm`** (same structural idea as **`EnginePluginResourceManager::_pm`**).
- **Static path configuration (optional mirror of engine):** methods such as **`setHeuristicPluginPaths` / `getHeuristicPluginPaths`** with the same **loading mode** semantics as **`EnginePluginResourceManager::setPluginPaths`** (absolute vs additive paths, **no path change while handles are active** unless the same restrictions as engine plugins apply).
- **`static std::shared_ptr<HeuristicPluginResourceManager> create()`** (or equivalent factory) builds the resource manager after constructing the **`HeuristicPluginManager`**.
- **Instance API (read-only selection):** for example:
  - **`createSelectionEngine(policyPluginId)`** — returns a **`std::unique_ptr<SelectionEngine>`** (or optional / nullable) for one loaded policy; the backend uses this in the outer loop; implementation uses [§8.6](#86-instance-lifecycle)–[§8.8](#88-selection-maps-to-sortengineids--sortedengineids).
  - **`resolveHeuristicPolicyOrder(descriptor, handle)`** (free function or member) — implements the precedence in [§5.3.3](#533-how-the-user-sets-orderedpolicyids) and returns **`orderedPolicyIds`** for **`finalize()`**.
  - **`getHeuristicPluginInfos()`** — optional, parallel to **`getEngineInfos()`** (plugin version, policy ID) for diagnostics.
  - **`getLoadedHeuristicPluginFiles(...)`** — optional, parallel to **`getLoadedPluginFiles`** on the engine resource manager.
- **Logging:** when heuristic `.so` files are loaded, call **`hipdnnHeuristicSetLoggingCallback`** (and optionally **`SetLogLevel`**) as defined in [§8.2](#82-plugin-module-metadata).

### 10.3 Handle integration

**Proposal:** **`hipdnnHandle`** exposes **`getHeuristicPluginResourceManager()`** alongside **`getPluginResourceManager()`**, returning a **`std::shared_ptr<HeuristicPluginResourceManager>`** created at handle construction (same era as the engine resource manager). If heuristic plugins are optional in early implementations, the pointer may refer to an empty manager that only supplies a **built-in** default `SelectionEngine` without loading external `.so` files.

### 10.4 Relationship to `EnginePluginResourceManager`

- **Candidate engine IDs** always come from **`EnginePluginResourceManager::getApplicableEngineIds`** (unchanged).
- **Ordering** is applied by **`SelectionEngine`** instances obtained from **`HeuristicPluginResourceManager`**.
- The two subsystems stay **separate**: heuristic plugins **do not** register engine IDs and **do not** execute graphs; engine plugins **do not** implement the heuristic C ABI.

---

## 11. Versioning and compatibility checks

Follow the same **spirit** as `EnginePluginManager::validateBeforeAdding` in the backend: reject incompatible plugins at load time with a clear error.

**Proposed checks**

1. **Heuristic C ABI major:** Parse **`hipdnnHeuristicGetApiVersion`**; **major** must match the backend’s expected heuristic API major (analogous to engine plugins comparing `plugin.apiVersion()` major to `HIPDNN_BACKEND_VERSION_MAJOR`, but using the **heuristic** version string, not the engine plugin API version).
2. **Policy ID uniqueness:** **`hipdnnHeuristicGetPolicyId`** must not collide with another loaded heuristic plugin’s policy ID.
3. **Binary compatibility:** Document minimum backend / data-SDK versions per heuristic plugin release (align with project-wide versioning RFCs under `docs/rfcs/`).

**On failure:** Do not register the plugin; log via the shared logging path ([§12](#12-logging)); continue loading other policies if policy loading is best-effort, or fail handle creation if strict mode is required (policy TBD).

**ABI evolution** is summarized in [§8.10](#810-abi-evolution).

---

## 12. Logging

### 12.1 Current state

Today, `hipdnnHandle` (**`struct hipdnnHandle`**) exposes stream and plugin resource manager functionality but **does not** expose a dedicated logger or `getLogger()` accessor. Backend code typically logs through **`hipdnn_backend::logging`** (`HIPDNN_BACKEND_LOG_*` macros in `backend/src/logging/Logging.hpp`), which ultimately dispatches via **`hipdnn_data_sdk::logging`** using the **global** user callback registered for the process.

Engine plugins may receive the logging callback through the existing plugin infrastructure (`PluginBase::setLoggingCallback`), keeping plugin logs on the **same** user-visible path as the backend.

### 12.2 How heuristic code obtains “the same logger” (same pattern as engine plugins)

Engine plugins do **not** take a logger on each API call. After the `.so` is loaded, **`PluginManagerBase::loadPluginFromFile`** calls **`plugin->setLoggingCallback(logging::backendLoggingCallback)`** once, then **`setLogLevel`**, so the plugin **stores** the callback and uses it internally (`backend/src/plugin/PluginCore.hpp`).

**Heuristic plugins should follow the same model:**

1. **Backend (C++) code** in the heuristic path uses **`HIPDNN_BACKEND_LOG_*`** / the same global SDK dispatch as today—no logger argument on **`SelectionEngine`** methods ([§7](#7-selectionengine-interface)).
2. **Heuristic `.so` code:** Immediately after loading a heuristic library (in **`HeuristicPluginManager`** / resource manager, mirroring **`loadPluginFromFile`**), the host calls **`hipdnnHeuristicSetLoggingCallback`** ([§8.2](#82-plugin-module-metadata)) with the same **`logging::backendLoggingCallback`** (or an equivalent that forwards to the consumer’s registered path), then optionally **`hipdnnHeuristicSetLogLevel`**. Individual C ABI entry points in [§8.6](#86-instance-lifecycle)–[§8.8](#88-selection-maps-to-sortengineids--sortedengineids) **do not** take a logging parameter.
3. **C ABI:** Only the module-level **`SetLoggingCallback`** / **`SetLogLevel`** symbols carry logging configuration—**not** the per-instance selection functions.

This matches engine plugins: **supplied and used outside the call signatures** of selection/engine operations, via one-time registration at load time.

---

## 13. Serialized graph and graph-level preferences

### 13.1 Serialized graph

`GraphDescriptor` already maintains a **FlatBuffer** serialized graph and exposes it via **`getSerializedGraph()`** (pointer and byte length). The heuristic framework should treat that buffer as the canonical **`SerializedGraph`** input to policies—**no second serialization format** in v1.

Policies that need structured access may parse the FlatBuffer using existing data-SDK generated types, subject to version rules for the graph schema. The C ABI passes this buffer as **`const uint8_t*`** + **`size_t`** ([§8.7](#87-configuration-and-applicability), [§8.8](#88-selection-maps-to-sortengineids--sortedengineids)).

### 13.2 Graph-level preferences (for example `preferred_engine_id`)

The graph model already carries fields such as **`preferred_engine_id`** when built from operation descriptors. **This draft assigns responsibility as follows:** interpreting graph-level preferences (honor, override, validate) is the job of a **concrete policy**—typically a **rule-based** or **config**-style selector—not a separate hard-coded pass in `EngineHeuristicDescriptor`. The outer loop may place that policy early in the ordered list so user intent affects ordering before ML or cache policies.

---

## 14. EngineHeuristicDescriptor integration

### 14.1 Responsibilities

`EngineHeuristicDescriptor` continues to:

- Own the operation graph reference and heuristic mode attributes (see [§15](#15-public-api-notes)).
- Discover **candidate** engines via **`EnginePluginResourceManager`**.
- Expose results through **`HIPDNN_ATTR_ENGINEHEUR_RESULTS`**.

Additionally it:

- Obtains **`HeuristicPluginResourceManager`** from the handle (proposal: **`getHeuristicPluginResourceManager()`**) and uses it to **instantiate** `SelectionEngine` objects per policy ID (backed by the C ABI in [§8](#8-c-abi-for-heuristic-plugins)).
- Resolves **`DeviceProperties`** (override or `queryDeviceProperties()`).
- Obtains serialized graph bytes from the finalized graph descriptor.
- Runs the **outer policy loop** described in [§5.1](#51-single-orchestration-model-outer-loop), using **`orderedPolicyIds`** from [§5.3](#53-ordered-policy-list-default-and-user-configuration).
- Stores the final ordered engine IDs for result construction.

### 14.2 Pseudocode for `finalize()` (first draft)

```text
finalize():
  validate descriptor state (graph set, mode set, not already finalized)

  handle    = graph.getHandle()
  engineRm  = handle.getPluginResourceManager()
  heurRm    = handle.getHeuristicPluginResourceManager()  // proposal: parallel to engineRm

  candidates = engineRm.getApplicableEngineIds(graph)

  devProps = userDeviceOverride if set else queryDeviceProperties()

  serializedGraph = graph.getSerializedGraph()  // ptr + size; graph must be usable for heuristics

  orderedPolicyIds = resolveHeuristicPolicyOrder(thisDescriptor, handle)
    // §5.3: descriptor attr > handle > env > default
    // { "SelectionEngine::Config", "SelectionEngine::Fallback" } if no override

  success = false
  for each policyId in orderedPolicyIds:
    selection = heurRm.createSelectionEngine(policyId)  // wraps C ABI from §8; null if unknown
    if selection is empty:
      continue
    selection.setConfig(devProps)
    if not selection.isApplicable(serializedGraph):
      continue
    working = copy(candidates)
    if selection.sortEngineIds(working, serializedGraph):
      candidates = selection.sortedEngineIds()
      success = true
      break

  if not success:
    fallbackSortEngineIds(candidates)  // mirrors today's utilities::sortEngineIds

  store candidates as _engineIds
  mark finalized
```

---

## 15. Public API notes

- **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER` (proposal):** Ordered list of policy ID strings for **`EngineHeuristicDescriptor`**, overriding handle/env defaults ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). Attribute type: array of strings (or equivalent) consistent with other vector attributes in the backend.
- **Handle-level override (proposal):** Extension API or **`HeuristicPluginResourceManager`** method to set the default policy order for all heuristic descriptors on that handle unless the descriptor sets **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER`**.
- **`HIPDNN_HEURISTIC_POLICY_ORDER` (optional env):** Comma-separated policy IDs; lowest precedence among user overrides ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)).
- **`HIPDNN_ATTR_ENGINEHEUR_MODE`:** Today the backend supports a narrow heuristic mode surface. This RFC does **not** remove the attribute; a future mapping might define default **policy order** per mode, or deprecate mode once **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER`** and handle defaults are sufficient. That decision is left open in this draft.
- **`HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`:** Proposed as the user-facing override for [§6.3](#63-proposed-override-descriptor-level-device-properties) when the descriptor type and setters are implemented.
- **No requirement for new enums** per new policy: adding a policy is **deployment + registry order** (policy IDs from **`hipdnnHeuristicGetPolicyId`**), not necessarily a new public enum value. Well-known IDs **`SelectionEngine::Config`** and **`SelectionEngine::Fallback`** are **strings**, not enum members.
- **Headers:** Publish **`HeuristicsPluginApi.h`** (name TBD) in **plugin_sdk** (or a sibling package) containing the types and declarations in [§8](#8-c-abi-for-heuristic-plugins), without including **engine** plugin API headers.

---

## 16. Testing

- **Unit tests** for each policy with synthetic `DeviceProperties` and small FlatBuffer graphs (no GPU required where possible).
- **Regression test** asserting the **default** policy matches current `utilities::sortEngineIds` ordering for a fixed candidate list.
- **Integration tests** with real graphs and devices when GPU is available.
- **ABI / loader tests** that load a minimal mock heuristic `.so`, verify **`hipdnnHeuristicGetApiVersion`** / **`Create`** / **`Destroy`**, and reject wrong major versions.
- **Policy order tests** that assert default **`{ "SelectionEngine::Config", "SelectionEngine::Fallback" }`**, descriptor override wins over handle, and unknown IDs are handled per policy.

---

## 17. Risks and open questions

- **Policy list syntax:** Comma-separated env vars and string array attributes need clear rules for embedded commas and empty tokens; validate unknown policy IDs strictly or leniently (skip vs error).
- **Failure modes:** If the cache or ML policy returns partial or invalid engine IDs, should the backend validate against `candidates` before accepting success?
- **Async selection:** When implemented, thread-safety of `SelectionEngine` and **`hipdnnHeuristicEngine_t`**, and lifetime of `SerializedGraph` buffers, must be specified.
- **`PluginApi.h` comment:** Today’s reference to heuristic plugins implementing `PluginApi.h` + `HeuristicsPluginApi.h` should be updated in code/docs to match **this** RFC: heuristic plugins use **only** the heuristic C ABI header, not **`PluginApi.h`**.

---

## 18. Glossary

| Term | Meaning |
|------|--------|
| **Engine plugin** | Shared library providing engines and execution; **distinct C ABI** from heuristic plugins. |
| **Heuristic / selection policy plugin** | Shared library implementing one outer-loop selection strategy via the C ABI in [§8](#8-c-abi-for-heuristic-plugins). |
| **Heuristic C ABI** | `extern "C"` symbol set: module metadata, **`hipdnnHeuristicEngine_t`** lifecycle, selection functions ([§8](#8-c-abi-for-heuristic-plugins)). |
| **HeuristicPluginManager** | Loads and validates heuristic `.so` files; analogous to **EnginePluginManager** but **heuristic-only** symbols. |
| **HeuristicPluginResourceManager** | Handle-scoped facade for heuristic plugins; **`createSelectionEngine`**, paths; analogous to **EnginePluginResourceManager**. |
| **Outer loop** | Ordered list of policies; first applicable successful policy wins. |
| **orderedPolicyIds** | Resolved policy ID strings for one **`finalize()`**; default **`{ "SelectionEngine::Config", "SelectionEngine::Fallback" }`** ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). |
| **DeviceProperties** | Explicit struct of device facts passed into selection; no HIP calls inside policies. |
| **SelectionEngine** | C++ facade over **`hipdnnHeuristicEngine_t`** for one policy run; stateful API reserved for future async results. |
| **SerializedGraph** | FlatBuffer bytes + length from the operation graph descriptor. |
