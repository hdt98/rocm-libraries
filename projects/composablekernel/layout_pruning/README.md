# Layout Pruning Project

This directory contains analysis and planning for splitting the Composable Kernel convolution library by data layout.

## Quick Start

**Problem**: MIOpen currently links against a monolithic CK convolution library (727 files), but only uses 3 data layouts (471 files). This causes unnecessarily long build times.

**Solution**: Split the library into 12 layout-specific libraries, allowing MIOpen to link only what it needs (35% reduction).

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This file - project overview |
| `PR_ANALYSIS.md` | Analysis of upstream PRs #3010 and #2099 |
| `IMPLEMENTATION_PLAN.md` | **⭐ START HERE** - Detailed implementation plan |
| `LAYOUT_FILE_MANIFEST.md` | Complete listing of all 727 files by target library |
| `all_convolution_files.txt` | Raw list of all convolution source files |
| `layout_mapping.json` | Machine-readable categorization data |
| `categorize_files.py` | Script that generated the categorization |

## Current Status

**Phase 1: Complete** ✅
- All 727 convolution files inventoried
- 100% categorized into 12 target libraries
- Categorization verified

**Phase 2: Ready to Start** ⏳
- Directory structure creation
- Awaiting approval to proceed

## Key Numbers

- **Total convolution files**: 727
- **Target libraries**: 12
- **MIOpen needs**: 3 libraries (471 files)
- **Potential savings**: 35% fewer files to build/link
- **Expected build time improvement**: 20-30%

## Related PRs

- [composable_kernel#3010](https://github.com/ROCm/composable_kernel/pull/3010) - Upstream split (closed)
- [rocm-libraries#2099](https://github.com/ROCm/rocm-libraries/pull/2099) - MIOpen integration (closed, to be resurrected)

## Target Libraries

| Library | Files | MIOpen? |
|---------|-------|---------|
| `device_conv2d_nhwgc_operations` | 226 | ✅ Yes |
| `device_conv3d_ndhwgc_operations` | 290 | ✅ Yes |
| `device_conv_old_operations` | 22 | ✅ Yes |
| `device_conv2d_gnhwc_operations` | 30 | No |
| `device_conv2d_ngchw_operations` | 49 | No |
| `device_conv3d_gndhwc_operations` | 25 | No |
| `device_conv3d_ngcdhw_operations` | 38 | No |
| `device_conv1d_gnwc_operations` | 10 | No |
| `device_conv1d_nwgc_operations` | 3 | No |
| `device_conv3d_nhwgc_operations` | 2 | No |
| `device_convnd_generic_operations` | 24 | No |
| `device_quantization_operations` | 8 | No |

## Next Steps

1. Review `IMPLEMENTATION_PLAN.md`
2. Approve Phase 2 approach
3. Execute directory creation
4. Begin file migration

## Questions?

Contact: brockhargreaves-amd (based on PR #2099 resurrection commit)

---

**Project Start**: 2026-03-25
**Current Phase**: 1 (Complete)
**Next Milestone**: Phase 2 approval
