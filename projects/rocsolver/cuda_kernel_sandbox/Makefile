# rocSOLVER Kernel Sandbox Makefile
#
# Build rocSOLVER kernels for CUDA to test with Compute Sanitizer
#
# Usage:
#   make                      # Build all sandboxes
#   make sanitizer-racecheck  # Run racecheck on getf2_small
#   make clean                # Clean build artifacts

NVCC ?= nvcc

# Debug flags for sanitizer support
# -g: host debug info
# -G: device debug info (required for racecheck)
# -lineinfo: line number info in device code
NVCC_FLAGS = -O0 -g -G -lineinfo --std=c++17

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
SANDBOXES = sandbox_getf2_small

# Default target
all: $(SANDBOXES)

# Build rules
sandbox_getf2_small: sandbox_getf2_small.cu \
                     kernels/getf2_small_kernel.cuh \
                     cuda_compat.cuh \
                     rocsolver_types.cuh \
                     device_helpers.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o $@ $<

# Clean
clean:
	rm -f $(SANDBOXES)

# Compute Sanitizer targets
# Note: These require CUDA Toolkit with compute-sanitizer installed

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

# Run all sanitizer checks
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

# Help
help:
	@echo "rocSOLVER Kernel Sandbox Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all                  - Build all sandbox executables"
	@echo "  clean                - Remove built executables"
	@echo ""
	@echo "Sanitizer targets:"
	@echo "  sanitizer-memcheck   - Run memory error detection"
	@echo "  sanitizer-racecheck  - Run race condition detection"
	@echo "  sanitizer-racecheck-all - Racecheck with full hazard reporting"
	@echo "  sanitizer-initcheck  - Run uninitialized memory detection"
	@echo "  sanitizer-synccheck  - Run synchronization error detection"
	@echo "  sanitizer-all        - Run all sanitizer checks"
	@echo ""
	@echo "Custom test targets:"
	@echo "  test-custom          - Run with custom params: make test-custom M=32 N=16 BATCH=4"
	@echo "  sanitizer-custom     - Run racecheck with custom params"
	@echo ""
	@echo "Variables:"
	@echo "  NVCC                 - NVCC compiler path (default: nvcc)"
	@echo "  CUDA_ARCH            - Target CUDA architectures"
	@echo "  M, N, BATCH          - Matrix parameters for custom tests"

.PHONY: all clean help \
        sanitizer-memcheck sanitizer-racecheck sanitizer-racecheck-all \
        sanitizer-initcheck sanitizer-synccheck sanitizer-all \
        test-custom sanitizer-custom
