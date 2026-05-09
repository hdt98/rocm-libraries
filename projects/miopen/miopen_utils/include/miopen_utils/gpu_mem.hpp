// Forwarding header — GPUMem is defined in driver/driver.hpp.
// This allows test code to include GPUMem without directly depending
// on the driver/ directory. The GPUMem class should eventually be
// extracted into a standalone header here.
#ifndef GUARD_MIOPEN_UTILS_GPU_MEM_HPP
#define GUARD_MIOPEN_UTILS_GPU_MEM_HPP

// Phase 1: Forward to driver.hpp which defines GPUMem.
// Phase 2: Extract GPUMem into this file directly.
#include "../../driver/driver.hpp"

#endif
