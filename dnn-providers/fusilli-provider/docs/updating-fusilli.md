# Updating fusilli and IREE

This document covers how to update the IREE/fusilli pin used by fusilli-provider.

## Background: why fusilli's pin is special

The rocm-libraries CI setup creates a circular dependency when fusilli ships a breaking API change:

fusilli-provider builds against fusilli and IREE. The sources come in via submodules in TheRock's `iree-libs` directory. rocm-libraries' CI checks out TheRock, which checks out the `iree-libs/fusilli` and `iree-libs/iree` submodules at the SHAs TheRock pins, and then builds fusilli-provider against them.

- TheRock's fusilli pin can't move forward until rocm-libraries' fusilli-provider is updated.
- rocm-libraries' fusilli-provider can't be updated to the new APIs until TheRock provides the new fusilli, because CI builds fusilli-provider against TheRock's pinned fusilli.

## How updates work today (with `dep_overrides.json`)

`rocm-libraries/dep_overrides.json` is the source of truth for the fusilli SHA used by rocm-libraries' CI.
The CI passes `--override-file dep_overrides.json` to TheRock's `fetch_sources.py`, which switches TheRock's `iree-libs/fusilli` and `iree-libs/iree` checkouts to the SHAs listed in that file.

TheRock's own CI uses the regular `.gitmodules` pins without overrides. To keep the two views in sync PRs that bump rocm-libraries in TheRock must *propagate* `dep_overrides.json` into TheRock's pointers each time it bumps the rocm-libraries pin; the auto-bump workflow in TheRock does this automatically.

### To update IREE/fusilli (no breaking change)

1. Open a PR to rocm-libraries that bumps the `fusilli` and `iree` entry in `dep_overrides.json` to the new SHA. The IREE SHA should be the equivalent of the version number pinned in fusilli's [`version.json`](https://github.com/iree-org/fusilli/blob/main/version.json). Fusilli should work with a wide variety of IREE versions, but in practice it's best to use the version fusilli has already tested and verified against.
2. CI runs against the new fusilli/IREE. If green, merge.
3. Within ~12 h, TheRock's submodule-bump workflow opens a TheRock PR that propagates the override (single PR atomically updates the `rocm-libraries`, `iree-libs/fusilli`, and `iree-libs/iree` pointers).

### To update IREE/fusilli with a breaking API change

Same process, but open a single rocm-libraries PR that simultaneously bumps the `fusilli`/`iree` entry in `dep_overrides.json`, **and** updates fusilli-provider source code for the new fusilli API.

## Alternatives to `dep_overrides.json` considered

### Alternative A: ship a fusilli patch in rocm-libraries

1. **rocm-libraries PR**: introduce the breaking changes plus a checked-in **fusilli patch** similar to TheRock's `patches/amd-mainline/` but with a location in rocm-libraries.
2. *Wait* for the rocm-libraries auto-bump to land in TheRock.
3.  **TheRock PR**: bump fusilli to the new SHA and adds a flag (env var, config option, sentinel) to make the build ignore the patch shipped from rocm-libraries.
4. *Wait* for the TheRock back-bump to land in rocm-libraries.
5. **rocm-libraries PR**: remove the patch.
6. **TheRock PR**: remove the temporary "ignore patch" flag added in step 3.

### Alternative B: disable `THEROCK_ENABLE_FUSILLI_PLUGIN` during the rollover

1. **TheRock PR**: bump `iree-libs/fusilli` to the new (breaking) SHA and disable fusilli-provider build.
2. *Wait* for the next TheRock back-bump in rocm-libraries. This updates rocm-libraries' pinned TheRock ref to the SHA with the new fusilli (which won't actually build).
3. **rocm-libraries PR**: update fusilli-provider source for the new fusilli API.
4. *Wait* for the next TheRock auto-bump. TheRock's `rocm-libraries` pointer moves to #3's SHA.
8. **TheRock PR**: re-enable the plugin build. It _should_ be green at this point, but at the same time this is the first time it will actually be tested...

### Why we built `dep_overrides.json` instead

`dep_overrides.json` collapses both alternatives into **one rocm-libraries PR with no explicit waits**. The fusilli SHA bump and the fusilli-provider source update land together; rocm-libraries CI builds the new combination immediately; the auto-bump catches up TheRock automatically.

## Reference

- Override mechanism: TheRock's `build_tools/fetch_sources.py` `--override-file` flag, `load_submodule_overrides`, `apply_submodule_overrides`.
- Propagation (auto-bump side): TheRock's `build_tools/github_actions/bump_automation.py`.
- **Source-of-truth convention**: iree/fusilli pins in TheRock are effectively *owned* by `rocm-libraries/dep_overrides.json`. The next auto-bump will undo any direct iree/fusilli pin update in TheRock that isn't matched by a corresponding override change. **Don't bump iree/fusilli pins directly in TheRock**. Bump them in `dep_overrides.json` and let the next auto-bump propagate.
