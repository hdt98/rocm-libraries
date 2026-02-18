# rocSOLVER Kernel Sandbox Makefile
#
# Build rocSOLVER kernels for CUDA to test with Compute Sanitizer
#
# Usage:
#   make                      # Build all sandboxes
#   make sanitizer-racecheck  # Run racecheck on getf2_small
#   make stebz-racecheck      # Run racecheck on stebz
#   make clean                # Clean build artifacts

NVCC ?= nvcc

# Debug flags for sanitizer support
# -g: host debug info
# -G: device debug info (required for racecheck)
# -lineinfo: line number info in device code
NVCC_FLAGS = -O0 -g -G --std=c++17

# CUDA architectures (adjust for your GPU)
# Ampere: sm_80, sm_86
# Ada Lovelace: sm_89
# Hopper: sm_90
CUDA_ARCH ?= -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86

# Include paths
INCLUDES = -I. -I./kernels

# All sandbox executables
SANDBOXES = sandbox_getf2_small sandbox_stebz sandbox_stedc_solve

# Default target
all: $(SANDBOXES)

# Build rules
sandbox_getf2_small: sandbox_getf2_small.cu \
                     kernels/getf2_small_kernel.cuh \
                     cuda_compat.cuh \
                     rocsolver_types.cuh \
                     device_helpers.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o $@ $<

sandbox_stebz: sandbox_stebz.cu \
               kernels/stebz_kernels.cuh \
               cuda_compat.cuh \
               rocsolver_types.cuh \
               device_helpers.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o $@ $<

sandbox_stedc_solve: sandbox_stedc_solve.cu \
                     kernels/stedc_solve_kernels.cuh \
                     cuda_compat.cuh \
                     rocsolver_types.cuh \
                     device_helpers.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o $@ $<

# Clean
clean:
	rm -f $(SANDBOXES)

###############################################################################
# getf2_small Sanitizer Targets
###############################################################################

# Memory error detection
sanitizer-memcheck: sandbox_getf2_small
	compute-sanitizer --tool memcheck ./sandbox_getf2_small

# Race condition detection (most relevant for missing __syncthreads)
sanitizer-racecheck: sandbox_getf2_small
	compute-sanitizer --tool racecheck ./sandbox_getf2_small

# More aggressive racecheck with hazard reporting
sanitizer-racecheck-all: sandbox_getf2_small
	compute-sanitizer --tool racecheck --racecheck-report all ./sandbox_getf2_small

# Uninitialized memory detection
sanitizer-initcheck: sandbox_getf2_small
	compute-sanitizer --tool initcheck ./sandbox_getf2_small

# Synchronization error detection
sanitizer-synccheck: sandbox_getf2_small
	compute-sanitizer --tool synccheck ./sandbox_getf2_small

# Run all sanitizer checks on getf2_small
sanitizer-all: sanitizer-memcheck sanitizer-racecheck sanitizer-initcheck sanitizer-synccheck

# Custom test with parameters
# Usage: make test-custom M=32 N=16 BATCH=4
M ?= 32
N ?= 16
BATCH ?= 4

test-custom: sandbox_getf2_small
	./sandbox_getf2_small $(M) $(N) $(BATCH)

sanitizer-custom: sandbox_getf2_small
	compute-sanitizer --tool racecheck ./sandbox_getf2_small $(M) $(N) $(BATCH)

###############################################################################
# STEBZ Sanitizer Targets
###############################################################################

STEBZ_N ?= 64
STEBZ_BATCH ?= 1
STEBZ_RANGE ?= all

# Run stebz
run-stebz: sandbox_stebz
	./sandbox_stebz $(STEBZ_N) $(STEBZ_BATCH) $(STEBZ_RANGE)

# Memory error detection for stebz
stebz-memcheck: sandbox_stebz
	compute-sanitizer --tool memcheck ./sandbox_stebz $(STEBZ_N) $(STEBZ_BATCH) $(STEBZ_RANGE)

# Race condition detection for stebz
stebz-racecheck: sandbox_stebz
	compute-sanitizer --tool racecheck ./sandbox_stebz $(STEBZ_N) $(STEBZ_BATCH) $(STEBZ_RANGE)

# More aggressive racecheck for stebz
stebz-racecheck-all: sandbox_stebz
	compute-sanitizer --tool racecheck --racecheck-report all ./sandbox_stebz $(STEBZ_N) $(STEBZ_BATCH) $(STEBZ_RANGE)

# Uninitialized memory detection for stebz
stebz-initcheck: sandbox_stebz
	compute-sanitizer --tool initcheck ./sandbox_stebz $(STEBZ_N) $(STEBZ_BATCH) $(STEBZ_RANGE)

# Synchronization error detection for stebz
stebz-synccheck: sandbox_stebz
	compute-sanitizer --tool synccheck ./sandbox_stebz $(STEBZ_N) $(STEBZ_BATCH) $(STEBZ_RANGE)

# Run all sanitizer checks on stebz
stebz-all: stebz-memcheck stebz-racecheck stebz-initcheck stebz-synccheck

###############################################################################
# STEDC Solve (slaed4) Sanitizer Targets
###############################################################################

STEDC_N ?= 16
STEDC_BATCH ?= 1
STEDC_REF ?= 0

# Run stedc_solve
run-stedc-solve: sandbox_stedc_solve
	./sandbox_stedc_solve $(STEDC_N) $(STEDC_BATCH) $(STEDC_REF)

# Memory error detection for stedc_solve
stedc-memcheck: sandbox_stedc_solve
	compute-sanitizer --tool memcheck ./sandbox_stedc_solve $(STEDC_N) $(STEDC_BATCH) $(STEDC_REF)

# Race condition detection for stedc_solve
stedc-racecheck: sandbox_stedc_solve
	compute-sanitizer --tool racecheck ./sandbox_stedc_solve $(STEDC_N) $(STEDC_BATCH) $(STEDC_REF)

# More aggressive racecheck for stedc_solve
stedc-racecheck-all: sandbox_stedc_solve
	compute-sanitizer --tool racecheck --racecheck-report all ./sandbox_stedc_solve $(STEDC_N) $(STEDC_BATCH) $(STEDC_REF)

# Uninitialized memory detection for stedc_solve
stedc-initcheck: sandbox_stedc_solve
	compute-sanitizer --tool initcheck ./sandbox_stedc_solve $(STEDC_N) $(STEDC_BATCH) $(STEDC_REF)

# Synchronization error detection for stedc_solve
stedc-synccheck: sandbox_stedc_solve
	compute-sanitizer --tool synccheck ./sandbox_stedc_solve $(STEDC_N) $(STEDC_BATCH) $(STEDC_REF)

# Run all sanitizer checks on stedc_solve
stedc-all: stedc-memcheck stedc-racecheck stedc-initcheck stedc-synccheck

###############################################################################
# Help
###############################################################################

help:
	@echo "rocSOLVER Kernel Sandbox Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all                  - Build all sandbox executables"
	@echo "  sandbox_getf2_small  - Build getf2_small sandbox"
	@echo "  sandbox_stebz        - Build stebz sandbox"
	@echo "  sandbox_stedc_solve  - Build stedc solve (slaed4) sandbox"
	@echo "  clean                - Remove built executables"
	@echo ""
	@echo "getf2_small Sanitizer targets:"
	@echo "  sanitizer-memcheck   - Run memory error detection"
	@echo "  sanitizer-racecheck  - Run race condition detection"
	@echo "  sanitizer-racecheck-all - Racecheck with full hazard reporting"
	@echo "  sanitizer-initcheck  - Run uninitialized memory detection"
	@echo "  sanitizer-synccheck  - Run synchronization error detection"
	@echo "  sanitizer-all        - Run all sanitizer checks"
	@echo ""
	@echo "STEBZ Sanitizer targets:"
	@echo "  run-stebz            - Run stebz sandbox"
	@echo "  stebz-memcheck       - Run memory error detection"
	@echo "  stebz-racecheck      - Run race condition detection"
	@echo "  stebz-racecheck-all  - Racecheck with full hazard reporting"
	@echo "  stebz-initcheck      - Run uninitialized memory detection"
	@echo "  stebz-synccheck      - Run synchronization error detection"
	@echo "  stebz-all            - Run all sanitizer checks on stebz"
	@echo ""
	@echo "STEDC Solve (slaed4) Sanitizer targets:"
	@echo "  run-stedc-solve      - Run stedc solve sandbox"
	@echo "  stedc-memcheck       - Run memory error detection"
	@echo "  stedc-racecheck      - Run race condition detection"
	@echo "  stedc-racecheck-all  - Racecheck with full hazard reporting"
	@echo "  stedc-initcheck      - Run uninitialized memory detection"
	@echo "  stedc-synccheck      - Run synchronization error detection"
	@echo "  stedc-all            - Run all sanitizer checks on stedc"
	@echo ""
	@echo "Custom test targets:"
	@echo "  test-custom          - Run getf2 with params: make test-custom M=32 N=16 BATCH=4"
	@echo "  sanitizer-custom     - Run racecheck with custom params"
	@echo ""
	@echo "Variables:"
	@echo "  NVCC                 - NVCC compiler path (default: nvcc)"
	@echo "  CUDA_ARCH            - Target CUDA architectures"
	@echo "  M, N, BATCH          - getf2_small matrix parameters"
	@echo "  STEBZ_N, STEBZ_BATCH, STEBZ_RANGE - stebz parameters"
	@echo "  STEDC_N, STEDC_BATCH, STEDC_REF - stedc solve parameters (REF=1 for slaed4)"

.PHONY: all clean help \
        sanitizer-memcheck sanitizer-racecheck sanitizer-racecheck-all \
        sanitizer-initcheck sanitizer-synccheck sanitizer-all \
        test-custom sanitizer-custom \
        run-stebz stebz-memcheck stebz-racecheck stebz-racecheck-all \
        stebz-initcheck stebz-synccheck stebz-all \
        run-stedc-solve stedc-memcheck stedc-racecheck stedc-racecheck-all \
        stedc-initcheck stedc-synccheck stedc-all
