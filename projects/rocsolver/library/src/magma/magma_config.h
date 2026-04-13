/* Minimal MAGMA configuration for hipSOLVER integration.
 * This replaces the CMake-generated magma_config.h from a full MAGMA build.
 * Only the settings needed for the geev and sytrs implementations are defined.
 */

#ifndef MAGMA_CONFIG_H
#define MAGMA_CONFIG_H

/* Use HIP backend (AMD GPUs via ROCm) */
#define MAGMA_HAVE_HIP

/* Fortran name mangling: append underscore (most common convention) */
#define ADD_

/* Use 32-bit integers for MAGMA (matching LAPACK default) */
/* #undef MAGMA_ILP64 */

/* Disable thread affinity (avoid conflicts with hipsolver's threading) */
#define MAGMA_NOAFFINITY

#endif /* MAGMA_CONFIG_H */
