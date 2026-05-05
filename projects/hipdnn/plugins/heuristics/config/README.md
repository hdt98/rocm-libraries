# Config Heuristic Plugin

## Overview

The **Config** heuristic plugin implements configuration-based engine selection
by interpreting user-supplied preferences. It resolves a preferred engine ID
from two sources and reorders the candidate list so that ID comes first.

## Purpose

Resolves a preferred engine ID from the following sources, in priority order:

1. `Graph.preferred_engine_id` — set via the frontend
   `Graph::set_preferred_engine_id_ext()` setter and packed into the serialized
   FlatBuffer graph.
2. A rule from the JSON file pointed to by `HIPDNN_ENGINE_OVERRIDE_FILE` that
   matches the first conv-like node (`conv_fprop`, `conv_dgrad`, `conv_wgrad`)
   in the graph. The file is parsed once and cached for the lifetime of the
   process.

If a preferred ID is resolved and present in the candidate list, the plugin
reorders the candidates so the preferred ID comes first while preserving the
relative order of the remaining IDs. Otherwise the plugin returns
`NOT_APPLICABLE` so the next policy in the chain runs.

It serves as:
1. **First policy in default order** - user preferences take precedence
2. **Graph-level preference interpreter** - implements RFC 0007 Section 13.3
3. **User control mechanism** - allows applications to guide engine selection
   either programmatically (the setter) or out-of-band (the JSON config file)

## Well-Known Policy

**Policy Name**: `SelectionHeuristic::Config`
**Policy ID**: `0xABC123...` (computed via FNV-1a hash of name)

This is a well-known policy listed FIRST in the default policy order:
```cpp
std::vector<std::string> defaultPolicies = {
    "SelectionHeuristic::Config",           // Try graph preferences first
    "SelectionHeuristic::StaticOrdering"    // Fallback to static ordering
};
```

## C ABI Implementation

Implements the full heuristic C ABI from RFC 0007 Section 8 (same as StaticOrdering).

## Algorithm

```cpp
void finalize() {
    // Parse serialized graph FlatBuffer
    auto graph = GetGraph(serializedGraphBuffer);

    // 1. Explicit setter wins over the override file.
    std::optional<int64_t> preferredEngineId = graph->preferred_engine_id();

    // 2. Otherwise consult HIPDNN_ENGINE_OVERRIDE_FILE for a rule whose
    //    op + tensor pattern matches the first conv-like node in the graph.
    if (!preferredEngineId) {
        if (auto config = EngineOverrideConfig::loadFromEnv()) {
            preferredEngineId = matchOverrideConfig(graph, *config);
        }
    }

    if (!preferredEngineId || not in candidates) {
        return NOT_APPLICABLE;  // Policy declines
    }

    // Reorder: preferred first, others maintain relative order
    sortedEngineIds = [*preferredEngineId] + (candidates - *preferredEngineId);

    return SUCCESS;
}
```

## Applicability

Config applies **conditionally**:
- ✅ **Applies** if a preferred engine ID is resolved AND it is in the
  candidate list. The ID may come from either:
  - `Graph.preferred_engine_id` (set via the frontend setter), or
  - A matched rule from `HIPDNN_ENGINE_OVERRIDE_FILE`.

- ❌ **Not applicable** (`returns 0`) if:
  - No serialized graph provided
  - Empty candidate list
  - Neither source produces a preferred engine ID
  - A preferred ID was resolved but is not in the candidate list

## Graph Schema

Uses the `preferred_engine_id` field from `graph.fbs`:

```flatbuffers
table Graph {
    name: string;
    compute_data_type: DataType;
    intermediate_data_type: DataType;
    io_data_type: DataType;
    tensors: [TensorAttributes];
    nodes: [Node];
    preferred_engine_id: long = null;  // <-- This field
}
```

## Override File Format

`HIPDNN_ENGINE_OVERRIDE_FILE` points to a JSON file with one or more
`engine_overrides` entries. Each entry pins an `engine_name` to a specific
operation when the listed `tensors` patterns match. A `dim` of `-1` is a
per-slot wildcard; the optional `stride` is matched the same way.

```json
{
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] },
        { "dim": [64, 3, 7, 7] }
      ]
    },
    {
      "op": "conv_fprop",
      "engine_name": "FUSILLI_ENGINE",
      "tensors": [
        { "dim": [-1, -1, -1, -1] },
        { "dim": [-1, -1, -1, -1] }
      ]
    }
  ]
}
```

Rules are evaluated in declaration order; the first match wins. Internally the
matcher partitions rules into an exact-dim hash bucket and an order-preserving
wildcard list per op, reconciled by declaration index.

## Example Usage

**Application code (using the frontend setter)**:
```cpp
hipdnn_frontend::Graph graph;
// ... build operations ...

graph.set_preferred_engine_id_ext(MIOPEN_ENGINE_ID);

graph.finalize();
auto engineConfigs = graph.getEngineConfigs();
// engineConfigs[0] will be MIOPEN_ENGINE (if applicable)
```

**Out-of-band override (no code changes)**:
```bash
export HIPDNN_ENGINE_OVERRIDE_FILE=/path/to/overrides.json
./your_app   # conv_fprop nodes matching the rule will prefer the named engine
```

**Result**:
- **Without Config plugin**: Engines ordered by StaticOrdering
- **With Config plugin**: Preferred engine first, then StaticOrdering for rest

## Build

```bash
cd projects/hipdnn/build
cmake -GNinja ..
ninja hipdnn_heuristic_config
```

The plugin is built as:
- **Library**: `libhipdnn_heuristic_config.so`
- **Install path**: `${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/heuristics/`
- **Build copy**: Automatically copied to `build/lib/hipdnn_plugins/heuristics/` for testing

## Testing

Unit tests for the matcher live in `tests/TestEngineOverrideConfig.cpp` and
build into `hipdnn_heuristic_config_tests`. End-to-end coverage of the
preferred-engine path is in `IntegrationGraphEngineFiltering` under the
frontend integration suite.

## Dependencies

- **hipdnn_plugin_sdk**: Provides C ABI types
- **hipdnn_data_sdk**: Provides utilities and `engineNameToId`
- **hipdnn_flatbuffers_sdk**: Graph FlatBuffer schema and FlatBuffers headers
- **nlohmann_json**: Parses the `HIPDNN_ENGINE_OVERRIDE_FILE` JSON config

## Version Compatibility

- **Plugin API version**: 1.0.0
- **Minimum backend version**: Requires RFC 0007 implementation
- **Schema compatibility**: Requires `graph.fbs` with `preferred_engine_id` field

## Logging

Logs are emitted through the backend logging callback:
- `PolicyFinalize: reordered N engines with preferred engine 0x... (from <source>) first` — on success, where `<source>` is either `graph.preferred_engine_id` or `HIPDNN_ENGINE_OVERRIDE_FILE`
- `PolicyFinalize: preferred engine 0x... (from <source>) not in candidates - not applicable` — when the resolved ID is missing from the candidate list
- `PolicyFinalize: no preferred engine resolved - not applicable` — when neither source produced an ID
- Errors logged at `HIPDNN_SEV_ERROR`

## RFC Reference

This plugin implements requirements from:
- **RFC 0007 Section 5.3**: Well-known policy names and IDs
- **RFC 0007 Section 9**: Policy plugins and outer loop
- **RFC 0007 Section 13.1**: Serialized graph input
- **RFC 0007 Section 13.3**: Graph-level preferences interpretation (critical requirement)
- **RFC 0007 Section 8**: C ABI for heuristic plugins

## Design Notes

**Why Config runs before StaticOrdering:**
- User preferences should take precedence over hard-coded defaults
- Config can decline (not applicable), allowing fallback to StaticOrdering
- Ensures user intent is respected when specified

**Why both the setter and the env-var live in the same plugin:**
- Both are user-supplied preferences that map to the same downstream action
  (move a chosen engine ID to the front of the candidate list).
- Keeping the override-file logic out of the frontend lets the same rules
  apply to any caller of the backend, not only frontend users.
- The setter wins over the override file so explicit application code is
  never silently overridden by a stale config on disk.

**Why Config is separate from backend logic:**
- RFC 0007 Section 13.3 explicitly assigns preference interpretation to policies
- Keeps backend generic and extensible
- Allows future preference-based policies (cache-aware, ML-based, etc.)
