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
5. [Design overview](#5-design-overview) (includes [5.4 Two-tier plugin objects](#54-two-tier-plugin-objects-handle-vs-policy-descriptor))
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

Device capabilities enter the system **only as explicit data** (`DeviceProperties`), not by probing HIP from inside plugins. Callers may later supply overrides via backend attributes (for example `HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`), aligned with the broader direction of **explicit device facts at the selection boundary** (supplied via **`hipdnnHeuristicHandleSetDeviceProperties`** on the plugin session handle—see [§8](#8-c-abi-for-heuristic-plugins)).

Heuristic plugins implement a **two-tier** **stable C ABI** ([§8](#8-c-abi-for-heuristic-plugins)): a long-lived **plugin handle** (**`hipdnnHeuristicHandle_t`**, **createHandle** / **destroyHandle** / **setDeviceProperties**) created with the same timing and storage pattern as other hipDNN plugin handles, and a **policy descriptor** (**`hipdnnHeuristicPolicyDescriptor_t`**, **createPolicyDescriptor** / **destroyPolicyDescriptor** / **setEngineIds** / **Finalize** / **getSortedIds**) whose **lifecycle is owned by** **`EngineHeuristicDescriptor`**—one per slot in the ordered policy list, destroyed with the heuristic descriptor. That ABI is **separate from the engine plugin ABI**—a `.so` is one or the other, never both.

The **`SelectionEngine`** C++ type (or equivalent facade) wraps a **policy descriptor** and forwards to the C ABI; **stateful tracking** in the plugin lives behind the **plugin handle**. **HeuristicPluginManager** (and handle-scoped **HeuristicPluginResourceManager**) own discovery, loading, version validation, registration, and **plugin-handle** instances per **`hipdnnHandle`**. **`EngineHeuristicDescriptor::finalize()`** walks the **ordered policy list** using **descriptor-owned** policy objects, which is **user-configurable** with default **`{ "SelectionEngine::Config", "SelectionEngine::StaticOrdering" }`** ([§5.3](#53-ordered-policy-list-default-and-user-configuration)).

There is **no** separate post-loop step that applies **`utilities::sortEngineIds`** (or any other ordering) inside the backend. Legacy deterministic ordering is available **only** when a policy in **`orderedPolicyIds`** implements it—for example **`SelectionEngine::StaticOrdering`**. If every policy declines or the list is misconfigured so that no policy succeeds, **`finalize()`** fails using the **same error path as the rest of the hipDNN backend** (for example **`THROW_IF_FALSE`** / **`HipdnnException`** with an appropriate **`hipdnnStatus_t`** such as **`HIPDNN_STATUS_INTERNAL_ERROR`**, matching other descriptor **`finalize()`** failures when a required step does not complete—exact status TBD).

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

This RFC preserves **equivalent deterministic ordering** as one **policy** (the “StaticOrdering” selector), implemented as a plugin or thin wrapper, **only when that policy appears** in **`orderedPolicyIds`**. It is **not** applied as unconditional backend logic after the outer loop.

---

## 4. Goals and non-goals

**Goals**

- **Outer-loop-only** orchestration: try an ordered list of **policy** plugins until one succeeds.
- **Explicit `DeviceProperties`** at the selection boundary; plugins do not call `hipGetDevice` / `hipGetDeviceProperties` themselves.
- **Pluggable policies** loaded through a **`HeuristicPluginManager`** / **`HeuristicPluginResourceManager`** layer that mirrors **engine** plugin *mechanics* (paths, discovery, version checks, optional static path configuration) but uses the **heuristic-only** C ABI in [§8](#8-c-abi-for-heuristic-plugins).
- **Stable C ABI** for heuristic `.so` files: module metadata, **plugin-handle** lifecycle, **policy-descriptor** lifecycle, and selection entry points as in [§8](#8-c-abi-for-heuristic-plugins) (to be codified in headers such as **`HeuristicsPluginApi.h`**).
- **Aligned lifetimes:** **`hipdnnHeuristicHandle_t`** per loaded heuristic module per **`hipdnnHandle`** (with engine-plugin handles); **`hipdnnHeuristicPolicyDescriptor_t`** instances **owned by** **`EngineHeuristicDescriptor`** (one per policy slot), created when the policy list for that descriptor is established and destroyed with the descriptor.
- **Same logging path** as the rest of the backend (see [§12 Logging](#12-logging)).
- **Version compatibility** at load time (see [§11](#11-versioning-and-compatibility-checks)).
- **User-configurable policy order** with a documented **default** list ([§5.3](#53-ordered-policy-list-default-and-user-configuration)).
- **No hidden fallback ordering:** if no policy in **`orderedPolicyIds`** reports success, **`EngineHeuristicDescriptor::finalize()`** fails using the **normal backend mechanism** (`THROW_IF_*` macros / **`HipdnnException`** and a **`hipdnnStatus_t`**—see [§5.1](#51-single-orchestration-model-outer-loop) and [§14.2](#142-pseudocode-for-finalize-first-draft)); there is no second pass that sorts candidates inside the core library.

**Non-goals (this draft)**

- Inner “stage chains” inside a single policy (for example PerEngine → Config → StaticOrdering as mandatory sub-steps). Policy composition is expressed by **ordering multiple policy plugins** in the outer list, not by nested pipelines inside one policy plugin implementation.
- Mandating async selection in v1; the API is only **shaped** to allow it later (including **`Finalize`** + **`GetSortedEngineIds`** as a two-phase C selection API in [§8.9](#89-finalize-and-sorted-results)).

---

## 5. Design overview

### 5.1 Single orchestration model: outer loop

There is **one** primary control flow:

1. Build the list of **candidate engine IDs** from existing **engine** plugins (unchanged).
2. Obtain **device properties** (see [§6](#6-device-properties)) and the **serialized operation graph** (see [§13](#13-serialized-graph-and-graph-level-preferences)).
3. Resolve the **ordered list of policy plugin IDs** ([§5.3](#53-ordered-policy-list-default-and-user-configuration)).
4. Ensure **`EngineHeuristicDescriptor`** owns one **plugin policy descriptor** per slot (see [§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)); each slot binds to the **`hipdnnHeuristicHandle_t`** for that policy’s loaded module (from **`HeuristicPluginResourceManager`**).
5. For each slot in order: push **`DeviceProperties`** to the module’s plugin handle (**`SetDeviceProperties`**), **setEngineIds** and **serialized graph** on that slot’s policy descriptor, call **Finalize**; if the policy **wins**, read **getSortedIds** and **stop**; otherwise continue.
6. If no policy succeeds, **`finalize()`** **fails** using the same pattern as other backend descriptor logic errors (for example **`THROW_IF_FALSE(success, HIPDNN_STATUS_INTERNAL_ERROR, …)`** throwing **`HipdnnException`**, analogous to **`GraphDescriptor::finalize()`** when verification fails—exact message and status code are implementation details). There is **no** additional built-in sort after the loop.

There is **no** separate inner registry of sub-stages inside a single policy plugin in this design. If a team wants “config then static ordering” inside one deliverable, they ship **one** policy plugin that implements that sequence internally, or they register two entries in the **outer** list.

### 5.2 Policy examples (illustrative)

| Policy (illustrative name) | Role |
|----------------------------|------|
| **Cache selector** | Reuse a persisted ordering keyed by graph fingerprint and device properties. |
| **ML selector** | Score candidates using a model or cost function. |
| **Rule selector** | Deterministic rules over graph features and `DeviceProperties`. |
| **StaticOrdering-style selector** | Reproduce current `sortEngineIds` behavior when listed in **`orderedPolicyIds`**; users often place it **last** in the default list so behavior matches today unless they omit it. |

These are **examples**; the registry uses stable string IDs chosen by each plugin (and exposed via **`hipdnnHeuristicGetPolicyId`** in [§8](#8-c-abi-for-heuristic-plugins)).

### 5.3 Ordered policy list: default and user configuration

The outer loop needs an ordered list **`orderedPolicyIds`**: each element is a **string** that must match **`hipdnnHeuristicGetPolicyId`** for a loaded heuristic plugin (or a **built-in** policy registered under the same ID without a separate `.so`).

#### 5.3.1 Well-known policy IDs

This draft standardizes two logical policies using **well-known string IDs** (the `SelectionEngine::` prefix is a **naming convention** in the string itself, not C++ language linkage or a shared type with the C++ `SelectionEngine` class):

| Policy ID string | Role |
|------------------|------|
| **`SelectionEngine::Config`** | Applies user / graph configuration (for example honoring **`preferred_engine_id`**, env-based disables, future descriptor knobs). Typically runs **first** so explicit intent overrides later policies. |
| **`SelectionEngine::StaticOrdering`** | Deterministic ordering aligned with today’s **`utilities::sortEngineIds`** (for example MIOPEN preference, deterministic engine last). Typically runs **after** Config in the **default** list; it is **not** invoked by the backend outside **`orderedPolicyIds`**. |

Shipped heuristic plugins (or built-in adapters) **must** export these exact strings from **`hipdnnHeuristicGetPolicyId`** when they implement the Config and StaticOrdering behaviors.

#### 5.3.2 Default `orderedPolicyIds`

If the user does **not** override the list, the backend uses:

```text
{ "SelectionEngine::Config", "SelectionEngine::StaticOrdering" }
```

So **Config** is tried first; if it does not win the outer loop (not applicable or the policy **`finalize()`** does not report success), **StaticOrdering** runs. **StaticOrdering** is expected to succeed for valid candidate sets when implemented correctly; if **no** listed policy succeeds (for example the user omits **StaticOrdering** and other policies all decline), **`EngineHeuristicDescriptor::finalize()`** fails per [§5.1](#51-single-orchestration-model-outer-loop)—there is **no** backend fallback sort.

#### 5.3.3 How the user sets `orderedPolicyIds`

**Resolution order (highest precedence first):**

1. **Engine-heuristic descriptor (per finalize)** — User sets an explicit ordered list via a backend attribute (proposal: **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER`** or extension equivalent: array of UTF-8 policy ID strings). Applies only to that **`EngineHeuristicDescriptor`** instance.
2. **hipDNN handle (per handle)** — Optional API on **`HeuristicPluginResourceManager`** or a handle extension (proposal: **`hipdnnSetHeuristicPolicyOrder_ext(handle, ids, count)`** or C++ **`setDefaultHeuristicPolicyOrder`**). Used when the descriptor has no override.
3. **Process environment (optional)** — Proposal: **`HIPDNN_HEURISTIC_POLICY_ORDER`** as a comma-separated list of policy IDs, applied when neither descriptor nor handle supplied a list. Exact syntax and escaping TBD.
4. **Built-in default** — **`{ "SelectionEngine::Config", "SelectionEngine::StaticOrdering" }`** when nothing above applies.

The implementation should **validate** each ID against loaded/built-in policies (and log or skip unknown IDs per policy TBD).

#### 5.3.4 Relationship to `finalize()`

`EngineHeuristicDescriptor::finalize()` (or a helper on **`HeuristicPluginResourceManager`**) **merges** the resolution order above into a single **`std::vector<std::string> orderedPolicyIds`**, ensures **policy descriptor** objects exist **one-to-one** with that list (recreating them if the list changed since last setup), then runs the outer loop in [§14.2](#142-pseudocode-for-finalize-first-draft). If the loop completes without a successful policy, **`finalize()`** does not mark the descriptor finalized and exits via **`HipdnnException`** like other backend failures.

### 5.4 Two-tier plugin objects: handle vs policy descriptor

| Tier | C typedef (illustrative) | Lifetime | Holds |
|------|---------------------------|----------|--------|
| **Plugin handle** | **`hipdnnHeuristicHandle_t`** | Same pattern as other plugin handles on **`hipdnnHandle`**; created when the heuristic module is paired with the handle, destroyed with handle teardown | Plugin **session** state (caches, tuning, scratch); receives **`SetDeviceProperties`** |
| **Policy descriptor** | **`hipdnnHeuristicPolicyDescriptor_t`** | **Owned by** **`EngineHeuristicDescriptor`**; one per entry in **`orderedPolicyIds`**; **created** when the descriptor’s policy list is established (**implementation choice:** on attribute set, bind, or lazy at first **`finalize()`**) and **destroyed** with the heuristic descriptor | **Candidate engine IDs** + **serialized graph** for the current selection; **Finalize** / **GetSortedEngineIds** result |

**Threading:** Plugin handles are **single-thread only** (**not** thread-safe). Parallelism uses **multiple** **`hipdnnHandle`** instances (each with its own heuristic plugin handles) or host-side serialization (policy TBD). All calls for a given **`hipdnnHeuristicHandle_t`** and its dependent policy descriptors on that thread must follow this contract.

**Design-note mapping:** **createHandle** / **destroyHandle** / **setDeviceProperties(Handle\*)** → [§8.6](#86-plugin-handle-lifecycle); **createPolicyDescriptor** / **destroyPolicyDescriptor** / **setEngineIds** / **Finalize** / **getSortedIds** → [§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)–[§8.9](#89-finalize-and-sorted-results). Documents that used the term **engineDescriptor** for the plugin-side object mean **policy descriptor** here, not hipDNN’s **`EngineDescriptor`** for computational engines.

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

**Important:** Plugins **must not** call this helper; they consume only the `DeviceProperties` POD the host passes via **`hipdnnHeuristicHandleSetDeviceProperties`** ([§8.6](#86-plugin-handle-lifecycle)).

### 6.3 Proposed override: descriptor-level device properties

**First-draft proposal:** Extend the engine-heuristic descriptor so callers can optionally supply device properties explicitly, using the already-reserved attribute space (for example `HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP` and/or a dedicated `HIPDNN_BACKEND_DEVICEPROP_DESCRIPTOR` type when implemented).

**Semantics (proposal):**

- If the heuristic descriptor has **no** device-property override set, `EngineHeuristicDescriptor::finalize()` uses `queryDeviceProperties()` once.
- If the user sets device properties on the descriptor, **that** structure is used instead of querying HIP in finalize.

This keeps **one** pathway into heuristics (`DeviceProperties`) while allowing tests, multi-GPU hosts, and API parity with “device props, not handle” designs.

---

## 7. SelectionEngine interface

`SelectionEngine` is a **C++ facade** used by the backend for **one policy slot** on an **`EngineHeuristicDescriptor`**. It wraps an opaque **`hipdnnHeuristicPolicyDescriptor_t`** ([§8.7](#87-policy-descriptor-per-slot-graph--candidate-ids)) created with the **`hipdnnHeuristicHandle_t`** for that policy’s module. **Session state** (caches, etc.) lives in the plugin **behind the handle**, not in this wrapper.

Device properties are **not** set on the facade: the host calls **`hipdnnHeuristicHandleSetDeviceProperties`** on the plugin handle (typically once per **`finalize()`** attempt per module, or once per **`finalize()`** after resolving overrides—implementation detail) before **`Finalize`** on the policy descriptor.

```cpp
class SelectionEngine
{
public:
    // Candidate engine IDs from EnginePluginResourceManager; mirrors setEngineIds ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph)).
    void setEngineIds(const std::vector<int64_t>& engineIds);

    // FlatBuffer bytes + length from the operation graph ([§13](#13-serialized-graph-and-graph-level-preferences)).
    void setSerializedGraph(const SerializedGraph& serializedGraph);

    // Runs applicability + selection inside the plugin; true => policy won the outer loop ([§8.9](#89-finalize-and-sorted-results)).
    bool finalize();

    // Valid after finalize() returned true; mirrors GetSortedEngineIds.
    std::vector<int64_t> getSortedEngineIds();
};
```

**Notes**

- **Logging** matches **engine** plugins: not part of these method signatures; see [§12](#12-logging).
- **`finalize` + `getSortedEngineIds`** mirror the C ABI two-phase pattern so a future revision can perform async work in **`Finalize`** without changing names.
- **`EngineHeuristicDescriptor`** owns **`SelectionEngine`** (or equivalent) instances **one per** resolved policy slot; lifetimes match [§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor).

---

## 8. C ABI for heuristic plugins

This section defines the **stable C-language ABI** for heuristic (selection policy) shared libraries. It is **orthogonal** to the **engine** plugin ABI: a single `.so` is either an **engine** plugin or a **heuristic** plugin, **not both**. Heuristic libraries **do not** export `hipdnnPluginGetName`, `hipdnnPluginGetType`, or any **engine** entry points from `PluginApi.h` / `EnginePluginApi.h`—only the symbols below (final names live in **`HeuristicsPluginApi.h`** or equivalent).

### 8.1 Design principles

- All exported symbols use **C linkage** (`extern "C"` from C++ implementations).
- Versioning and rejection of incompatible plugins follow the same **ideas** as engine plugins (major API compatibility), but the **API version string** and **symbol set** are **heuristic-specific** ([§11](#11-versioning-and-compatibility-checks)).
- Plugins **must not** call HIP, mutate hipDNN graph descriptors, or execute engines; they only **read** inputs described here and **write** reordered engine IDs.
- The host (backend) owns output buffers unless the API explicitly transfers ownership.
- **Logging** is configured only via **`hipdnnHeuristicSetLoggingCallback`** / **`SetLogLevel`** at **module load time** ([§8.2](#82-plugin-module-metadata)); handle and policy-descriptor entry points do not take a logger argument ([§12](#12-logging)).

### 8.2 Plugin module metadata

Each heuristic `.so` exports the following (names are illustrative; implementations use the `hipdnnHeuristic` prefix in headers).

| Function | Purpose |
|----------|---------|
| `hipdnnHeuristicGetApiVersion(const char** version)` | Semantic version of **this C ABI** (for example `"1.0.0"`). Host rejects load on **major** mismatch. |
| `hipdnnHeuristicGetPolicyId(const char** policy_id)` | Stable string ID matched against the ordered policy list (for example **`SelectionEngine::StaticOrdering`** or a vendor-specific ID; see [§5.3](#53-ordered-policy-list-default-and-user-configuration)). |
| `hipdnnHeuristicGetPluginVersion(const char** version)` | Plugin implementation version (informational). |
| `hipdnnHeuristicSetLoggingCallback(hipdnnCallback_t cb)` | Registers the consumer logging callback; optional `hipdnnHeuristicSetLogLevel(hipdnnSeverity_t)` mirroring engine plugin behavior. |
| `hipdnnHeuristicGetLastErrorString(const char** msg)` | Per-thread last error after a failed call; pointer valid only for immediate use (same contract as `hipdnnPluginGetLastErrorString`). |

The host identifies heuristic libraries by **which loader** opened them and by successful resolution of **`hipdnnHeuristicGetApiVersion`** (and related required symbols), not by reusing **`hipdnnPluginGetType`**.

### 8.3 Plugin handle (session object)

Each loaded heuristic module exposes a **plugin handle**: an opaque **session** object. The host creates **one** **`hipdnnHeuristicHandle_t`** per **loaded heuristic `.so`** associated with a given **`hipdnnHandle`**, using the **same timing and storage pattern** as other hipDNN plugin handles, so every future selection path can **forward that plugin-specific handle** and **stateful tracking** can live in the plugin behind it.

**Threading contract:** A plugin handle is **not** thread-safe (**single-thread only**). Concurrent use requires **separate** **`hipdnnHandle`** instances (and thus separate heuristic plugin handles) or host-side serialization (policy TBD).

```c
typedef struct hipdnnHeuristicHandle_opaque* hipdnnHeuristicHandle_t;
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

### 8.6 Plugin handle lifecycle

Illustrative names (headers may shorten or alias symbols—exact spelling TBD):

```c
hipdnnHeuristicStatus_t hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle);
hipdnnHeuristicStatus_t hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle);

hipdnnHeuristicStatus_t hipdnnHeuristicHandleSetDeviceProperties(
    hipdnnHeuristicHandle_t handle,
    const hipdnnHeuristicDeviceProperties_t* props);
```

- **`Create`** / **`Destroy`** correspond to **createHandle** / **destroyHandle**: the host invokes them when binding a heuristic module to a **`hipdnnHandle`** (alongside other plugin-handle setup).
- **`SetDeviceProperties`** corresponds to **setDeviceProperties(Handle\*)**: the host supplies explicit POD only; plugins **must not** call HIP. The backend uses **`queryDeviceProperties()`** or descriptor override ([§6](#6-device-properties)) and forwards the result (typically when driving **`finalize()`**).

### 8.7 Policy descriptor (per-slot graph + candidate IDs)

A **policy descriptor** is a second opaque object. It holds **per–`EngineHeuristicDescriptor` slot** state: **candidate engine IDs**, **serialized graph** bytes, and internal state through **`Finalize`**.

**Naming:** This document uses **policy descriptor** in the C ABI to avoid confusion with hipDNN’s **`EngineDescriptor`** (computational engine). Informal design notes that say **engineDescriptor** for the plugin-side object refer to **policy descriptor** here.

```c
typedef struct hipdnnHeuristicPolicyDescriptor_opaque* hipdnnHeuristicPolicyDescriptor_t;
```

**Lifecycle:** The backend **`EngineHeuristicDescriptor` owns** one **`hipdnnHeuristicPolicyDescriptor_t`** per entry in **`orderedPolicyIds`**. Objects are **created** when the descriptor’s policy list is established (on attribute set, bind, or lazily at first **`finalize()`**—implementation choice) and **destroyed** when the **`EngineHeuristicDescriptor`** is destroyed (**plugin policy descriptors and the hipDNN heuristic descriptor die together**).

```c
hipdnnHeuristicStatus_t hipdnnHeuristicPolicyDescriptorCreate(
    hipdnnHeuristicHandle_t plugin_handle,
    hipdnnHeuristicPolicyDescriptor_t* out_desc);

hipdnnHeuristicStatus_t hipdnnHeuristicPolicyDescriptorDestroy(
    hipdnnHeuristicPolicyDescriptor_t desc);
```

### 8.8 Policy inputs: engine IDs and serialized graph

```c
hipdnnHeuristicStatus_t hipdnnHeuristicPolicySetEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc,
    const int64_t* engine_ids,
    size_t engine_id_count);

hipdnnHeuristicStatus_t hipdnnHeuristicPolicySetSerializedGraph(
    hipdnnHeuristicPolicyDescriptor_t desc,
    const uint8_t* serialized_graph,
    size_t serialized_graph_size);
```

These correspond to **setEngineIds** and to storing **graph details** on the plugin policy descriptor.

### 8.9 Finalize and sorted results

**Two-phase** selection (matches **`SelectionEngine::finalize`** / **`getSortedEngineIds`**; leaves room for future async **`Finalize`**):

```c
/* *out_applied == 1 => policy won; host then calls GetSortedEngineIds.
   *out_applied == 0 => not applicable or declined; host continues outer loop. */
hipdnnHeuristicStatus_t hipdnnHeuristicPolicyFinalize(
    hipdnnHeuristicPolicyDescriptor_t desc,
    int32_t* out_applied);

hipdnnHeuristicStatus_t hipdnnHeuristicPolicyGetSortedEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc,
    int64_t* out_ids,
    size_t out_capacity,
    size_t* out_count);
```

**Contract**

- Calls on a policy descriptor that hang off a given **`hipdnnHeuristicHandle_t`** must occur on a **thread consistent** with that handle’s **single-thread** contract ([§8.3](#83-plugin-handle-session-object)).
- **`SetEngineIds` / `SetSerializedGraph` / `Finalize`:** candidate IDs come from **`EnginePluginResourceManager`**; output IDs **must** be a **permutation or subset** of the **SetEngineIds** input (host validates).
- **`GetSortedEngineIds`:** valid only after **`Finalize`** with **`out_applied == 1`**.

### 8.10 Host integration (C++ backend)

**`HeuristicPlugin`** resolves **both** handle and policy symbols via `dlsym` (or equivalent).

**`HeuristicPluginResourceManager`** (per **`hipdnnHandle`**):

1. After loading each heuristic `.so`, calls **`hipdnnHeuristicSetLoggingCallback`** (and optionally **`SetLogLevel`**)—same timing as engine plugins ([§12](#12-logging)).
2. Creates **`hipdnnHeuristicHandle_t`** via **`hipdnnHeuristicHandleCreate`** for each accepted module, stores it **like other plugin handles**, and exposes lookup (for example **`getHeuristicHandleForPolicyId`**) for the backend.
3. Does **not** own **`EngineHeuristicDescriptor`**-scoped policy descriptors; those are created with **`hipdnnHeuristicPolicyDescriptorCreate(plugin_handle, …)`** when the heuristic descriptor (re)builds its policy slot table.

**`EngineHeuristicDescriptor`** holds **`std::vector<std::unique_ptr<SelectionEngine>>`** (or equivalent): each **`SelectionEngine`** wraps one **`hipdnnHeuristicPolicyDescriptor_t`**, bound to the **`hipdnnHeuristicHandle_t`** for that slot’s policy module. **`finalize()`** resolves **`orderedPolicyIds`**, **SetDeviceProperties** on each distinct handle used, then for each slot **SetEngineIds** / **SetSerializedGraph** / **Finalize** / **GetSortedEngineIds** as in [§14.2](#142-pseudocode-for-finalize-first-draft). If no slot succeeds, **`finalize()`** aborts via **`HipdnnException`** (normal backend error path); there is **no** post-loop **`utilities::sortEngineIds`**.

### 8.11 ABI evolution

- **Patch/minor:** additive optional functions, or use of **`reserved`** fields with version negotiation.
- **Major:** breaking struct layout or required function set; incompatible plugins fail **`validateBeforeAdding`**-style checks at load time ([§11](#11-versioning-and-compatibility-checks)).

---

## 9. Policy plugins and the outer loop

Each **heuristic policy plugin** implements one selection strategy and is loaded by **`HeuristicPluginManager`** (see [§10](#10-heuristicpluginmanager-and-resource-layer)). The backend maintains an **ordered list** of **policy plugin IDs** (strings) and **owns** one **policy descriptor** + **`SelectionEngine`** wrapper **per slot** on **`EngineHeuristicDescriptor`** ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)). For each slot in order during **`finalize()`**, the backend:

1. Resolves the **`hipdnnHeuristicHandle_t`** for that slot’s module (from **`HeuristicPluginResourceManager`**); skips the slot if unknown / failed to load—policy TBD.
2. Calls **`hipdnnHeuristicHandleSetDeviceProperties`** on that handle with resolved **`DeviceProperties`**.
3. Calls **`setEngineIds`** and **`setSerializedGraph`** on the slot’s **`SelectionEngine`** (policy descriptor).
4. Calls **`finalize()`**; if false, **continue** with the **original** candidate engine list unchanged for the next slot.
5. On success, replaces candidates with **`getSortedEngineIds()`** and **breaks**.
6. On failure, continues; the next policy always starts from the **original** candidate list (unless a future RFC defines chaining semantics).

**First-success wins:** The first applicable policy that reports success defines the final ordering. Later policies are not consulted.

**Exhausted list:** If every slot is skipped, declined, or errors without a successful policy, **`EngineHeuristicDescriptor::finalize()`** fails with **`HipdnnException`** (same **`THROW_IF_*` / status pattern as other backend descriptor `finalize()` paths**—see [§5.1](#51-single-orchestration-model-outer-loop)); the implementation does **not** fall back to **`utilities::sortEngineIds`** or any other ordering outside **`orderedPolicyIds`**.

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
- Owns **`HeuristicPlugin`** wrappers that bind the C ABI (handle + policy symbols) and expose **`HandleCreate` / `HandleDestroy`** to the resource manager.

### 10.2 `HeuristicPluginResourceManager`

Analogous to **`EnginePluginResourceManager`**:

- Holds **`std::shared_ptr<HeuristicPluginManager> _pm`** (same structural idea as **`EnginePluginResourceManager::_pm`**).
- **Static path configuration (optional mirror of engine):** methods such as **`setHeuristicPluginPaths` / `getHeuristicPluginPaths`** with the same **loading mode** semantics as **`EnginePluginResourceManager::setPluginPaths`** (absolute vs additive paths, **no path change while handles are active** unless the same restrictions as engine plugins apply).
- **`static std::shared_ptr<HeuristicPluginResourceManager> create()`** (or equivalent factory) builds the resource manager after constructing the **`HeuristicPluginManager`**.
- **Instance API (read-only selection):** for example:
  - **`getHeuristicHandleForPolicyId(policyPluginId)`** — returns the stored **`hipdnnHeuristicHandle_t`** (or null) for that loaded module; created at handle setup via [§8.6](#86-plugin-handle-lifecycle).
  - **`resolveHeuristicPolicyOrder(descriptor, handle)`** (free function or member) — implements the precedence in [§5.3.3](#533-how-the-user-sets-orderedpolicyids) and returns **`orderedPolicyIds`** for **`finalize()`**.
  - **`getHeuristicPluginInfos()`** — optional, parallel to **`getEngineInfos()`** (plugin version, policy ID) for diagnostics.
  - **`getLoadedHeuristicPluginFiles(...)`** — optional, parallel to **`getLoadedPluginFiles`** on the engine resource manager.
- **Logging:** when heuristic `.so` files are loaded, call **`hipdnnHeuristicSetLoggingCallback`** (and optionally **`SetLogLevel`**) as defined in [§8.2](#82-plugin-module-metadata).

### 10.3 Handle integration

**Proposal:** **`hipdnnHandle`** exposes **`getHeuristicPluginResourceManager()`** alongside **`getPluginResourceManager()`**, returning a **`std::shared_ptr<HeuristicPluginResourceManager>`** created at handle construction (same era as the engine resource manager). At that time the manager creates and stores **`hipdnnHeuristicHandle_t`** values per loaded heuristic module ([§8.6](#86-plugin-handle-lifecycle)). If heuristic plugins are optional in early implementations, the pointer may refer to an empty manager that only supplies **built-in** handles/adapters without loading external `.so` files.

### 10.4 Relationship to `EnginePluginResourceManager`

- **Candidate engine IDs** always come from **`EnginePluginResourceManager::getApplicableEngineIds`** (unchanged).
- **Ordering** is applied by **`SelectionEngine`** instances **owned by** **`EngineHeuristicDescriptor`**, each wrapping a **`hipdnnHeuristicPolicyDescriptor_t`** bound to a **`hipdnnHeuristicHandle_t`** from **`HeuristicPluginResourceManager`**.
- The two subsystems stay **separate**: heuristic plugins **do not** register engine IDs and **do not** execute graphs; engine plugins **do not** implement the heuristic C ABI.

---

## 11. Versioning and compatibility checks

Follow the same **spirit** as `EnginePluginManager::validateBeforeAdding` in the backend: reject incompatible plugins at load time with a clear error.

**Proposed checks**

1. **Heuristic C ABI major:** Parse **`hipdnnHeuristicGetApiVersion`**; **major** must match the backend’s expected heuristic API major (analogous to engine plugins comparing `plugin.apiVersion()` major to `HIPDNN_BACKEND_VERSION_MAJOR`, but using the **heuristic** version string, not the engine plugin API version).
2. **Policy ID uniqueness:** **`hipdnnHeuristicGetPolicyId`** must not collide with another loaded heuristic plugin’s policy ID.
3. **Binary compatibility:** Document minimum backend / data-SDK versions per heuristic plugin release (align with project-wide versioning RFCs under `docs/rfcs/`).

**On failure:** Do not register the plugin; log via the shared logging path ([§12](#12-logging)); continue loading other policies if policy loading is best-effort, or fail handle creation if strict mode is required (policy TBD).

**ABI evolution** is summarized in [§8.11](#811-abi-evolution).

---

## 12. Logging

### 12.1 Current state

Today, `hipdnnHandle` (**`struct hipdnnHandle`**) exposes stream and plugin resource manager functionality but **does not** expose a dedicated logger or `getLogger()` accessor. Backend code typically logs through **`hipdnn_backend::logging`** (`HIPDNN_BACKEND_LOG_*` macros in `backend/src/logging/Logging.hpp`), which ultimately dispatches via **`hipdnn_data_sdk::logging`** using the **global** user callback registered for the process.

Engine plugins may receive the logging callback through the existing plugin infrastructure (`PluginBase::setLoggingCallback`), keeping plugin logs on the **same** user-visible path as the backend.

### 12.2 How heuristic code obtains “the same logger” (same pattern as engine plugins)

Engine plugins do **not** take a logger on each API call. After the `.so` is loaded, **`PluginManagerBase::loadPluginFromFile`** calls **`plugin->setLoggingCallback(logging::backendLoggingCallback)`** once, then **`setLogLevel`**, so the plugin **stores** the callback and uses it internally (`backend/src/plugin/PluginCore.hpp`).

**Heuristic plugins should follow the same model:**

1. **Backend (C++) code** in the heuristic path uses **`HIPDNN_BACKEND_LOG_*`** / the same global SDK dispatch as today—no logger argument on **`SelectionEngine`** or handle methods ([§7](#7-selectionengine-interface)).
2. **Heuristic `.so` code:** Immediately after loading a heuristic library (in **`HeuristicPluginManager`** / resource manager, mirroring **`loadPluginFromFile`**), the host calls **`hipdnnHeuristicSetLoggingCallback`** ([§8.2](#82-plugin-module-metadata)) with the same **`logging::backendLoggingCallback`** (or an equivalent that forwards to the consumer’s registered path), then optionally **`hipdnnHeuristicSetLogLevel`**. Handle and policy-descriptor entry points in [§8.6](#86-plugin-handle-lifecycle)–[§8.9](#89-finalize-and-sorted-results) **do not** take a logging parameter.
3. **C ABI:** Only the module-level **`SetLoggingCallback`** / **`SetLogLevel`** symbols carry logging configuration—**not** the per-instance selection functions.

This matches engine plugins: **supplied and used outside the call signatures** of selection/engine operations, via one-time registration at load time.

---

## 13. Serialized graph and graph-level preferences

### 13.1 Serialized graph

`GraphDescriptor` already maintains a **FlatBuffer** serialized graph and exposes it via **`getSerializedGraph()`** (pointer and byte length). The heuristic framework should treat that buffer as the canonical **`SerializedGraph`** input to policies—**no second serialization format** in v1.

Policies that need structured access may parse the FlatBuffer using existing data-SDK generated types, subject to version rules for the graph schema. The C ABI passes this buffer as **`const uint8_t*`** + **`size_t`** via **`hipdnnHeuristicPolicySetSerializedGraph`** ([§8.8](#88-policy-inputs-engine-ids-and-serialized-graph)).

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

- Obtains **`HeuristicPluginResourceManager`** from the handle (proposal: **`getHeuristicPluginResourceManager()`**) for **`hipdnnHeuristicHandle_t`** lookup per policy module.
- **Owns** **`SelectionEngine`** (policy-descriptor) objects **one per** resolved policy slot ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)); (re)creates them when **`orderedPolicyIds`** changes.
- Resolves **`DeviceProperties`** (override or `queryDeviceProperties()`).
- Obtains serialized graph bytes from the finalized graph descriptor.
- Runs the **outer policy loop** described in [§5.1](#51-single-orchestration-model-outer-loop), using **`orderedPolicyIds`** from [§5.3](#53-ordered-policy-list-default-and-user-configuration).
- On success, stores the final ordered engine IDs for result construction; on **exhausted list without success**, aborts **`finalize()`** via **`HipdnnException`** (normal backend error path—[§14.2](#142-pseudocode-for-finalize-first-draft)).

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
    // { "SelectionEngine::Config", "SelectionEngine::StaticOrdering" } if no override

  syncPolicySlots(thisDescriptor, orderedPolicyIds, heurRm)
    // Ensure one SelectionEngine (hipdnnHeuristicPolicyDescriptor_t) per slot, each bound to
    // the hipdnnHeuristicHandle_t for that policy's module; destroy/recreate if list changed.

  success = false
  for each slot i aligned with orderedPolicyIds:
    pluginHandle = heurRm.getHeuristicHandleForPolicyId(orderedPolicyIds[i])
    if pluginHandle is null:
      continue
    hipdnnHeuristicHandleSetDeviceProperties(pluginHandle, devProps)  // POD only; §8.6

    selection = thisDescriptor.policySlot(i)  // wraps policy descriptor; §8.7
    selection.setEngineIds(candidates)
    selection.setSerializedGraph(serializedGraph)
    if not selection.finalize():  // §8.9; not applicable or declined
      continue
    candidates = selection.getSortedEngineIds()
    success = true
    break

  if not success:
    // Abort: same as other descriptor finalize() failures — THROW_IF_FALSE / HipdnnException
    // with hipdnnStatus_t (e.g. HIPDNN_STATUS_INTERNAL_ERROR). Do not call utilities::sortEngineIds.
    return via exception; descriptor stays not finalized

  store candidates as _engineIds
  mark finalized
```

---

## 15. Public API notes

- **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER` (proposal):** Ordered list of policy ID strings for **`EngineHeuristicDescriptor`**, overriding handle/env defaults ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). Attribute type: array of strings (or equivalent) consistent with other vector attributes in the backend. When this list changes, the backend **recreates** the owned **policy descriptor** objects ([§5.4](#54-two-tier-plugin-objects-handle-vs-policy-descriptor)).
- **Handle-level override (proposal):** Extension API or **`HeuristicPluginResourceManager`** method to set the default policy order for all heuristic descriptors on that handle unless the descriptor sets **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER`**.
- **`HIPDNN_HEURISTIC_POLICY_ORDER` (optional env):** Comma-separated policy IDs; lowest precedence among user overrides ([§5.3.3](#533-how-the-user-sets-orderedpolicyids)).
- **`HIPDNN_ATTR_ENGINEHEUR_MODE`:** Today the backend supports a narrow heuristic mode surface. This RFC does **not** remove the attribute; a future mapping might define default **policy order** per mode, or deprecate mode once **`HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER`** and handle defaults are sufficient. That decision is left open in this draft.
- **`HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP`:** Proposed as the user-facing override for [§6.3](#63-proposed-override-descriptor-level-device-properties) when the descriptor type and setters are implemented.
- **No requirement for new enums** per new policy: adding a policy is **deployment + registry order** (policy IDs from **`hipdnnHeuristicGetPolicyId`**), not necessarily a new public enum value. Well-known IDs **`SelectionEngine::Config`** and **`SelectionEngine::StaticOrdering`** are **strings**, not enum members.
- **Headers:** Publish **`HeuristicsPluginApi.h`** (name TBD) in **plugin_sdk** (or a sibling package) containing the types and declarations in [§8](#8-c-abi-for-heuristic-plugins), without including **engine** plugin API headers.

---

## 16. Testing

- **Unit tests** for each policy with synthetic `DeviceProperties` and small FlatBuffer graphs (no GPU required where possible).
- **Regression test** asserting that when **`SelectionEngine::StaticOrdering`** is in effect (for example via the default **`orderedPolicyIds`**), ordering matches current `utilities::sortEngineIds` for a fixed candidate list.
- **Failure test** asserting that when **`orderedPolicyIds`** is empty, all IDs are unknown/skipped, or every policy declines, **`finalize()`** fails via **`HipdnnException`** / the same status path as other descriptor **`finalize()`** errors (no silent sort fallback).
- **Integration tests** with real graphs and devices when GPU is available.
- **ABI / loader tests** that load a minimal mock heuristic `.so`, verify **`hipdnnHeuristicGetApiVersion`**, **`HandleCreate` / `HandleDestroy`**, **`PolicyDescriptorCreate` / `Destroy`**, **`Finalize` / `GetSortedEngineIds`**, and reject wrong major versions.
- **Policy order tests** that assert default **`{ "SelectionEngine::Config", "SelectionEngine::StaticOrdering" }`**, descriptor override wins over handle, and unknown IDs are handled per policy.
- **Lifetime tests** that assert destroying **`EngineHeuristicDescriptor`** invokes **`hipdnnHeuristicPolicyDescriptorDestroy`** for every owned slot (and does not leak plugin handles owned by **`hipdnnHandle`**).

---

## 17. Risks and open questions

- **Policy list syntax:** Comma-separated env vars and string array attributes need clear rules for embedded commas and empty tokens; validate unknown policy IDs strictly or leniently (skip vs error). If skipping unknown IDs leaves no successful policy, **`finalize()`** fails (no backend fallback sort).
- **Duplicate policy IDs in `orderedPolicyIds`:** Two slots may reference the same loaded module (same **`hipdnnHeuristicHandle_t`**) but still require **distinct** **`hipdnnHeuristicPolicyDescriptor_t`** instances; whether duplicates are allowed, deduplicated, or rejected is policy TBD.
- **Failure modes:** If the cache or ML policy returns partial or invalid engine IDs, should the backend validate against `candidates` before accepting success?
- **Async selection:** When implemented, thread-safety of **`hipdnnHeuristicHandle_t`** / **`hipdnnHeuristicPolicyDescriptor_t`**, interaction with the **single-thread handle** rule ([§8.3](#83-plugin-handle-session-object)), and lifetime of `SerializedGraph` buffers, must be specified.
- **`PluginApi.h` comment:** Today’s reference to heuristic plugins implementing `PluginApi.h` + `HeuristicsPluginApi.h` should be updated in code/docs to match **this** RFC: heuristic plugins use **only** the heuristic C ABI header, not **`PluginApi.h`**.

---

## 18. Glossary

| Term | Meaning |
|------|--------|
| **Engine plugin** | Shared library providing engines and execution; **distinct C ABI** from heuristic plugins. |
| **Heuristic / selection policy plugin** | Shared library implementing one outer-loop selection strategy via the C ABI in [§8](#8-c-abi-for-heuristic-plugins). |
| **Heuristic C ABI** | `extern "C"` symbol set: module metadata, **`hipdnnHeuristicHandle_t`** and **`hipdnnHeuristicPolicyDescriptor_t`** lifecycle, selection functions ([§8](#8-c-abi-for-heuristic-plugins)). |
| **HeuristicPluginManager** | Loads and validates heuristic `.so` files; analogous to **EnginePluginManager** but **heuristic-only** symbols. |
| **HeuristicPluginResourceManager** | Handle-scoped facade for heuristic plugins; stores **`hipdnnHeuristicHandle_t`** per module, paths; analogous to **EnginePluginResourceManager**. |
| **Outer loop** | Ordered list of policies; first applicable successful policy wins. If none succeed, **`finalize()`** fails via **`HipdnnException`** (normal backend error path); no built-in sort after the loop. |
| **orderedPolicyIds** | Resolved policy ID strings for one **`finalize()`**; default **`{ "SelectionEngine::Config", "SelectionEngine::StaticOrdering" }`** ([§5.3](#53-ordered-policy-list-default-and-user-configuration)). |
| **DeviceProperties** | Explicit struct of device facts passed into selection; no HIP calls inside policies. |
| **SelectionEngine** | C++ facade over **`hipdnnHeuristicPolicyDescriptor_t`** for one policy **slot** on **`EngineHeuristicDescriptor`**; session state stays on **`hipdnnHeuristicHandle_t`**. |
| **Plugin heuristic handle** | **`hipdnnHeuristicHandle_t`**: session object per heuristic module per **`hipdnnHandle`**; **SetDeviceProperties**; **single-thread** use ([§8.3](#83-plugin-handle-session-object)). |
| **Plugin policy descriptor** | **`hipdnnHeuristicPolicyDescriptor_t`**: per-slot graph + candidate IDs + finalize result; **owned by** **`EngineHeuristicDescriptor`**. |
| **SerializedGraph** | FlatBuffer bytes + length from the operation graph descriptor. |
